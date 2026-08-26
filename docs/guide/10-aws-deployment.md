# Chapter 10: AWS Cloud Deployment

## 1. What "the cloud" actually is

Running an application on your own laptop means it's only reachable while your laptop is on and
connected. **Cloud computing** means renting compute, storage, and networking from a provider
(AWS, Azure, GCP) that runs enormous data centers on your behalf — you get a computer (or a
managed way to run a container, or a database, etc.) that's always on, reachable from the
internet, and billed by usage rather than owned outright. AWS (Amazon Web Services) is one such
provider, and it's what this project deploys to.

Everything below is explained from zero — no prior AWS knowledge assumed — and is a factual
account of what this project's actual deployment does and the real incidents hit building it, not
a generic AWS tutorial.

## 2. The core AWS building blocks this project uses

- **IAM (Identity and Access Management)** — AWS's permission system. Every action against AWS
  (start a container, read a secret, write a log) requires the caller to be an identity IAM
  recognizes, holding a policy that explicitly grants that specific action. Nothing is allowed by
  default. A **role** is an identity that something (a GitHub Actions workflow, a running
  container) can *assume* temporarily, rather than a permanent set of credentials tied to a
  person.
- **OIDC federation** — instead of storing a long-lived AWS access key as a GitHub secret (a
  standing credential that, if ever leaked, works forever until manually revoked), this project
  uses **OpenID Connect federation**: AWS trusts GitHub's own identity tokens directly. An IAM
  **OIDC provider** is configured to trust `token.actions.githubusercontent.com`, scoped to only
  this specific repository (`repo:Arun2198/enterprise-rag-platform:*`). When a GitHub Actions
  workflow run needs AWS access, it presents a short-lived token GitHub itself issued, AWS
  verifies it came from the trusted OIDC provider and matches the trusted repo, and grants
  temporary credentials for that one workflow run only. No AWS credential is ever stored as a
  GitHub secret at all.
- **ECR (Elastic Container Registry)** — AWS's managed storage for Docker images ([Chapter 9](
  09-containers-and-docker.md)). The CI pipeline builds the image and pushes it here; ECS then
  pulls from here to actually run it.
- **ECS (Elastic Container Service) Fargate** — AWS's managed way to run containers without
  managing the underlying servers yourself ("Fargate" specifically means serverless — you specify
  CPU/memory for your container and AWS handles provisioning the actual machine it runs on,
  as opposed to "EC2 launch type," where you'd manage a fleet of virtual machines yourself). A
  **cluster** is a logical grouping of services; a **service** keeps a specified number of
  container instances (**tasks**) running, restarting them if they crash.
- **ALB (Application Load Balancer)** — sits in front of the running containers, routes incoming
  HTTP traffic to a healthy task, and is what a client actually connects to (the containers
  themselves aren't directly reachable from the internet).
- **Bedrock** — AWS's managed API for calling foundation models (Claude, Titan, Llama, etc.)
  without hosting the model yourself. Used here as one of the two LLM generation providers
  ([Chapter 5](05-generation-and-llms.md)).

## 3. ECS Fargate "Express Mode" — why this project uses it

A normal ECS deployment requires you to separately define and wire together the load balancer,
target groups, a TLS certificate, security groups, and auto-scaling policies — real infrastructure
work with many places to misconfigure something. **Express Mode** (via the
`aws-actions/amazon-ecs-deploy-express-service` GitHub Action) auto-provisions all of that from a
much smaller set of inputs — this project's `.github/workflows/deploy-aws.yml` calls it with just
a service name, image, IAM roles, cluster, container port, environment variables, CPU/memory, and
scaling bounds:

```yaml
- name: Deploy to ECS Express Mode
  uses: aws-actions/amazon-ecs-deploy-express-service@v1
  with:
    service-name: ${{ env.ECS_SERVICE }}
    image: ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ env.IMAGE_TAG }}
    execution-role-arn: arn:aws:iam::...:role/ecsTaskExecutionRole
    task-role-arn: arn:aws:iam::...:role/ecsTaskRole
    infrastructure-role-arn: arn:aws:iam::...:role/ecsInfrastructureRoleForExpressServices
    cluster: ${{ env.ECS_CLUSTER }}
    container-port: 8000
    cpu: '1024'
    memory: '4096'
    health-check-path: /health
    min-task-count: 1
    max-task-count: 2
    auto-scaling-metric: AVERAGE_CPU
    auto-scaling-target-value: 70
```

Note three *separate* IAM roles, each with a distinct job:

- **`ecsTaskExecutionRole`** — used by ECS itself to pull the image from ECR and ship logs to
  CloudWatch. This is infrastructure plumbing, not the application's own permissions.
- **`ecsTaskRole`** — the permissions the *running application code itself* gets — in this
  project, this is what lets `BedrockAnswerer` actually call the Bedrock API from inside the
  container.
- **`ecsInfrastructureRoleForExpressServices`** — Express Mode's own provisioning identity, used
  to create the ALB, target groups, TLS certificate, and security groups on your behalf.

`min-task-count: 1` / `max-task-count: 2` with `auto-scaling-metric: AVERAGE_CPU` /
`auto-scaling-target-value: 70` means: always keep at least one task running, and if average CPU
usage across tasks exceeds 70%, scale up to a second task — simple, low-cost auto-scaling for a
demo-scale deployment.

## 4. Environment variables and secrets at deploy time

The same workflow's `environment-variables` block is exactly [Chapter 1](
01-project-overview.md#5-configuration)'s configuration system, populated for the live deployment:

```yaml
environment-variables: |
  [
    {"name": "RERANKER_ENABLED", "value": "true"},
    {"name": "EMBEDDING_PROVIDER", "value": "sentence_transformer"},
    {"name": "EMBEDDING_MODEL_NAME", "value": "BAAI/bge-base-en-v1.5"},
    {"name": "GENERATION_PROVIDER", "value": "${{ vars.GENERATION_PROVIDER }}"},
    {"name": "BEDROCK_MODEL_ID", "value": "arn:aws:bedrock:...inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"},
    {"name": "LLM_BASE_URL", "value": "https://models.github.ai/inference"},
    ...
  ]
secrets: |
  [
    {"name": "LLM_API_KEY", "valueFrom": "${{ vars.LLM_API_KEY_SECRET_ARN }}"}
  ]
```

Note the split: plain config values are passed as literal environment variables directly in the
workflow file, but `LLM_API_KEY` is passed as a **secret reference** (`valueFrom`, an ARN pointing
into AWS Secrets Manager) rather than a literal value anywhere in the workflow or its logs — ECS
resolves it at container-start time. `${{ vars.* }}` values come from **GitHub repository
variables** (`AWS_REGION`, `AWS_ACCOUNT_ID`, `ECR_REPOSITORY`, `ECS_CLUSTER`, `ECS_SERVICE`,
`GENERATION_PROVIDER`, `GENERATION_FALLBACK_PROVIDER`, `LLM_API_KEY_SECRET_ARN`) — non-secret
configuration that's still convenient to keep out of the workflow file itself, e.g. so the AWS
account id isn't hardcoded into source.

`BEDROCK_MODEL_ID` is set to a full **inference profile ARN**, not a plain model id — see
[Chapter 5](05-generation-and-llms.md#3-bedrockanswerer--aws-bedrock) for why: some newer Claude
models are only invocable through Bedrock via an inference profile.

## 5. Real production incidents hit while building this deployment

This isn't a clean, first-try deployment — everything below actually happened and was root-caused
using real evidence (CloudTrail event logs, the ECS console's Resources panel), not guesswork:

- **Missing ECS capacity providers.** A plain `aws ecs create-cluster` does **not** automatically
  attach the FARGATE/FARGATE_SPOT capacity providers a Fargate service needs to actually schedule
  tasks — this has to be done explicitly via `put-cluster-capacity-providers`. Missing this was the
  root cause of a multi-hour stuck deployment where tasks simply never started, with no obviously
  diagnostic error message pointing at the real cause.
- **Missing `iam:PassRole` for a newly-added role.** When `ecsTaskRole` was introduced (to give the
  running container Bedrock access), ECS itself needed permission to *hand that role to the task
  it was starting* — a distinct permission (`iam:PassRole`) from the role's own permissions.
  Forgetting to grant it produces a failure at task-launch time, not at the point where the role is
  actually used.
- **Missing `application-autoscaling:TagResource`.** A subtler gap than a typical "access denied":
  the auto-scaling target's own *registration* (`RegisterScalableTarget`) succeeded, but a
  *separate* call to tag that just-created resource failed for lack of a distinct permission — the
  kind of partial-success failure mode that's easy to misdiagnose as "it should have worked, the
  main permission is there."
- **Missing CloudWatch Logs / ACM certificate permissions** on
  `ecsInfrastructureRoleForExpressServices` — Express Mode provisions a TLS certificate and log
  group on your behalf, which means its own IAM role needs permission to create those things, not
  just the application's roles.
- **A Bedrock `AccessDeniedException` for `INVALID_PAYMENT_INSTRUMENT`** — this was neither an IAM
  problem nor a credit-balance problem: certain Bedrock foundation models require a **separate AWS
  Marketplace subscription** step, independent of both IAM permissions and account billing status.
  This is exactly the class of failure [Chapter 5](
  05-generation-and-llms.md#5-fallbackanswerer--provider-redundancy)'s `FallbackAnswerer` exists
  for — it's a real failure mode this deployment actually hit, not a hypothetical one.

**The most reliable debugging tools across all of these**: the ECS console's **Resources panel**
on the service page, and **CloudTrail's `lookup-events`** filtered by `EventName` — both
consistently surfaced the real cause faster than `aws ecs describe-services`, whose event log
stayed empty even during real, in-progress failures.

## 6. What's explicitly *not* hardened here

Documented honestly rather than hidden: **all four IAM roles in this deployment currently carry
AWS-managed `*FullAccess` policies, not least-privilege custom policies** — a deliberate, flagged
tradeoff for demo speed, never narrowed down. There's also no private VPC networking (the ALB and
tasks sit in default networking), no authentication on the API endpoints, no rate limiting, ECR
image scanning is disabled, and there are no CloudWatch alarms or dashboards configured. See
[Chapter 13](13-security-and-glossary.md) for the full, honest accounting of what's missing and
why each gap matters (or doesn't, for a demo-scale deployment).

## 7. Cost and teardown

Running continuously, this deployment costs roughly $55-60/month (ECS Fargate compute + ALB + data
transfer); a few hours of actual hands-on use costs cents. To stop paying while keeping the cheap-
to-recreate pieces (ECR images, IAM roles, the cluster definition itself) intact for a fast
redeploy later:

```bash
aws ecs update-service --cluster enterprise-rag-cluster --service enterprise-rag-service --desired-count 0 --region us-east-1
aws ecs delete-service --cluster enterprise-rag-cluster --service enterprise-rag-service --force --region us-east-1
```

Redeploying later is a single command (`gh workflow run deploy-aws.yml`, ~5-25 minutes) since all
the one-time IAM/ECR/cluster setup already exists.

Next: [Chapter 11 — CI/CD with GitHub Actions](11-cicd-and-github-actions.md).
