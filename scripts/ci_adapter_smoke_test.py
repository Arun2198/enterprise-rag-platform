"""
End-to-end smoke test for GitHubActionsDeploymentPipeline against a real
GitHub Actions workflow.

Not part of the pytest suite (unit tests mock subprocess/gh entirely and
must not dispatch real workflow runs) - run this directly from a machine
with the gh CLI installed and authenticated (`gh auth login`). Dispatches
a real workflow run, visible in the repo's Actions tab and consuming CI
minutes, so this is deliberately a manual/opt-in script rather than
something CI or the test suite runs on its own.

    uv run python scripts/ci_adapter_smoke_test.py

Override the repo/workflow/ref via env vars if needed:

    GITHUB_REPO=owner/name GITHUB_TEST_WORKFLOW=ci.yml GITHUB_REF=main \
        uv run python scripts/ci_adapter_smoke_test.py
"""
import os
import shutil
import subprocess
import sys

from mlops.deployment import GitHubActionsDeploymentPipeline

DEFAULT_REPO = "Arun2198/enterprise-rag-platform"
DEFAULT_WORKFLOW = "llm-e2e-smoke-test.yml"


def _check_gh_ready() -> str | None:
    if shutil.which("gh") is None:
        return "gh CLI not found on PATH - install it first (see README/CLAUDE.md for steps)."

    status = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)

    if status.returncode != 0:
        return "gh is installed but not authenticated - run `gh auth login` first."

    return None


def main() -> int:
    readiness_error = _check_gh_ready()

    if readiness_error:
        print(f"error: {readiness_error}", file=sys.stderr)
        return 1

    repo = os.getenv("GITHUB_REPO", DEFAULT_REPO)
    workflow = os.getenv("GITHUB_TEST_WORKFLOW", DEFAULT_WORKFLOW)
    ref = os.getenv("GITHUB_REF", "main")

    pipeline = GitHubActionsDeploymentPipeline(
        repo=repo,
        test_workflow=workflow,
        ref=ref
    )

    print(f"repo: {repo}")
    print(f"workflow: {workflow}")
    print(f"ref: {ref}")
    print("dispatching and polling for completion (this can take a couple of minutes)...")

    result = pipeline.run_tests()

    print(f"stage: {result.stage}")
    print(f"success: {result.success}")
    print(f"duration_seconds: {result.duration_seconds:.1f}")
    print(f"output: {result.output}")

    if not result.success:
        print("FAILED: see output above")
        return 1

    print("OK: workflow run completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
