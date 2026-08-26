# Chapter 11: CI/CD with GitHub Actions

## 1. What CI/CD means, from scratch

**Continuous Integration (CI)** is the practice of automatically running checks (tests, linters,
builds) every time code changes, instead of relying on someone to remember to run them manually.
**Continuous Deployment (CD)** goes further: automatically shipping a change to a real running
environment once it passes those checks, instead of a person manually building and uploading
something. Together, "CI/CD" is the automated pipeline between "a developer pushes code" and "the
change is live," with as few manual, error-prone steps as possible in between.

**GitHub Actions** is GitHub's built-in automation system: you define **workflows** as YAML files
in `.github/workflows/`, each triggered by some event (a push, a pull request, a manual button
click, a schedule), running a sequence of **jobs** made of **steps** on a fresh virtual machine
GitHub provisions for the run.

## 2. This project's workflows

`.github/workflows/` contains exactly two files:

- **`deploy-aws.yml`** — triggered on every push to `main`, plus a manual `workflow_dispatch`
  trigger. This is the one that actually builds and deploys the application, covered in full in
  [Chapter 10](10-aws-deployment.md#3-ecs-fargate-express-mode--why-this-project-uses-it).
- **`llm-e2e-smoke-test.yml`** — manual-only (`workflow_dispatch`), runs a real end-to-end check
  against the GitHub Models LLM endpoint via `scripts/llm_e2e_smoke_test.py`, using the
  repository's own built-in `GITHUB_TOKEN` (scoped with `models: read` permission) as the API key
  — no separately managed secret needed for this particular check, since GitHub Models
  authenticates with GitHub's own tokens.

**An honest, load-bearing fact worth stating plainly**: there is currently **no CI workflow that
runs the pytest suite automatically on push or pull request**. `deploy-aws.yml` triggers straight
from a push to `main` to building and deploying the Docker image — it does not run `uv run pytest`
as a gate beforehand. The test suite ([Chapter 12](12-testing-strategy.md)) is real, comprehensive,
and fast, but running it is currently a manual step (`uv run pytest`) a developer does before
pushing, not an automated gate GitHub enforces. This is a real gap, not a design choice being
defended — see [Chapter 13](13-security-and-glossary.md) for the full honest accounting of what's
intentionally deferred versus what's simply not built yet.

## 3. Tracing exactly what happens on a real `git push origin main`

Walking through `deploy-aws.yml` ([full contents in Chapter 10](
10-aws-deployment.md#3-ecs-fargate-express-mode--why-this-project-uses-it)) step by step, for a
concrete push:

1. **Trigger.** GitHub detects the push to `main` and schedules a new workflow run on a fresh
   Ubuntu virtual machine.
2. **`actions/checkout@v4`.** Clones the repository at the exact commit that was pushed onto that
   fresh machine — the workflow starts with nothing else present.
3. **`aws-actions/configure-aws-credentials@v5`.** This is the OIDC federation step described in
   [Chapter 10](10-aws-deployment.md#2-the-core-aws-building-blocks-this-project-uses): the
   workflow (which has `permissions: id-token: write` declared at the top of the file, without
   which GitHub won't let it request an OIDC token at all) requests a short-lived identity token
   from GitHub, presents it to AWS, and AWS — because it trusts GitHub's OIDC provider scoped to
   exactly this repository — hands back temporary credentials for the
   `github-actions-ecs-deploy-role`. No stored AWS access key is ever involved.
4. **`aws-actions/amazon-ecr-login@v2`.** Uses those temporary credentials to authenticate Docker
   against this account's ECR registry, so the next step is allowed to push there.
5. **Compute the image tag.** `IMAGE_TAG=${GITHUB_SHA:0:7}` — the first 7 characters of the commit
   hash that triggered this run become the image's version tag, alongside a floating `latest` tag.
   This means every deployed image is traceable back to the exact commit that produced it.
6. **`docker/build-push-action@v6`.** Builds the image from the repo's `Dockerfile`
   ([Chapter 9](09-containers-and-docker.md)) and pushes it to ECR under both tags.
7. **`aws-actions/amazon-ecs-deploy-express-service@v1`.** Tells ECS Express Mode to deploy this
   exact new image, with the environment variables and IAM roles from [Chapter 10](
   10-aws-deployment.md#4-environment-variables-and-secrets-at-deploy-time). ECS starts new
   task(s) running the new image, waits for them to pass the `/health` check, and only then shifts
   traffic away from any old tasks — this rolling-replacement behavior is what Express Mode's ALB
   integration provides, meaning a deploy doesn't cause a hard outage for requests in flight.

From "developer runs `git push`" to "new code is serving real traffic," roughly 5-25 minutes
elapse, entirely unattended once the push happens.

## 4. `[skip ci]` — a deliberate escape hatch

Because `deploy-aws.yml` triggers on *every* push to `main` with no test gate in front of it, a
push that isn't actually ready to go live (say, the required GitHub repository variables haven't
been configured yet, or a change is being staged before the AWS side is ready) would otherwise
trigger a deploy attempt regardless. Including the literal string `[skip ci]` anywhere in a commit
message is a GitHub-recognized convention that suppresses workflow triggers for that push
entirely. This project used it deliberately at specific points during development — a real,
intentional lever for "commit this now, but don't try to deploy it yet," not an accident or an
oversight.

## 5. The `GitHubActionsDeploymentPipeline` — a separate, more general piece

Worth distinguishing from the workflow files above: `src/mlops/deployment.py` defines a
`DeploymentPipeline` Protocol (`run_tests`/`run_evaluation`/`run_experiment`/`deploy`, each
returning a `StageResult`) as part of the MLOps platform ([Chapter 8](08-mlops-platform.md)).
`GitHubActionsDeploymentPipeline` is one implementation of it — it drives real GitHub Actions
workflows *programmatically*, through the `gh` CLI (`gh workflow run` to dispatch a run, `gh run
list` / `gh run view --json` to poll for it and read its conclusion), reusing whatever `gh auth
login` session or `GH_TOKEN`/`GITHUB_TOKEN` is already available in the calling environment rather
than rolling its own REST client and token handling. Each of its four stages maps to an
independently configurable workflow file name; a stage left unconfigured is a soft no-op rather
than forcing every repo using this class to have all four workflows defined.

This is **not** what actually deploys this project — `deploy-aws.yml` triggering directly on push
is what does that. `GitHubActionsDeploymentPipeline` is a general-purpose, reusable building block
for driving GitHub Actions from *other* orchestration code (for example, a hypothetical future
`ValidationWorkflow` from [Chapter 8](08-mlops-platform.md#6-the-rest-of-the-platform-built-tested-not-yet-wired-into-the-request-path)
that runs evaluation and only promotes/deploys if quality metrics hold up). It's also, honestly,
not verified against a real `gh` invocation in this repo's own development — it's covered by unit
tests with `subprocess.run` mocked out, not a live end-to-end run, since the sandbox this was
built in didn't have `gh` installed.

Next: [Chapter 12 — Testing Strategy](12-testing-strategy.md).
