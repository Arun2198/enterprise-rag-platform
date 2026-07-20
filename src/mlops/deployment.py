import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_TAIL_CHARS = 2000


@dataclass(frozen=True)
class StageResult:
    stage: str
    success: bool
    duration_seconds: float
    output: str = ""


@dataclass(frozen=True)
class DeploymentResult:
    pipeline: str
    stages: list[StageResult]

    @property
    def success(self) -> bool:
        return all(stage.success for stage in self.stages)


class DeploymentPipeline(Protocol):
    """
    test -> evaluate -> experiment -> deploy. Same shape whether it's
    driven by GitHub Actions (GitHubActionsDeploymentPipeline below),
    Azure DevOps, Jenkins, GitLab CI, or in-process
    (LocalDeploymentPipeline below) - a CI system's job is to call these
    stages in order and act on the result. Azure DevOps/Jenkins/GitLab CI
    adapters are still not implemented (each would call that provider's
    own REST API/CLI), but they'd implement this exact Protocol so
    PlatformManager and everything downstream is unaffected by which CI
    system actually runs it.
    """
    name: str

    def run_tests(self) -> StageResult:
        ...

    def run_evaluation(self) -> StageResult:
        ...

    def run_experiment(self, experiment_id: str) -> StageResult:
        ...

    def deploy(self, asset_id: str, target_stage: str) -> StageResult:
        ...


class LocalDeploymentPipeline:
    """
    Real, working reference implementation: runs the actual test suite
    and evaluation CLI in-process via subprocess, so DeploymentPipeline
    is provably a usable interface and not just a paper Protocol.
    `deploy()` here only records intent (via the logger) rather than
    doing anything - actually deploying is inherently provider-specific.
    """
    name = "local"

    def __init__(
        self,
        repo_root: Path | None = None
    ) -> None:
        self.repo_root = repo_root or REPO_ROOT

    def run_tests(self) -> StageResult:
        return self._run_stage("test", [sys.executable, "-m", "pytest", "-q"])

    def run_evaluation(self) -> StageResult:
        return self._run_stage(
            "evaluate",
            [
                sys.executable, "evaluation/run_eval.py",
                "--dataset", "evaluation/golden_dataset.json"
            ]
        )

    def run_experiment(
        self,
        experiment_id: str
    ) -> StageResult:
        # an experiment is just a named evaluation run with its own
        # output location - reuse the same eval CLI rather than a
        # separate experiment runner
        return self._run_stage(
            "experiment",
            [
                sys.executable, "evaluation/run_eval.py",
                "--dataset", "evaluation/golden_dataset.json",
                "--output", f"evaluation/reports/{experiment_id}",
                "--json"
            ]
        )

    def deploy(
        self,
        asset_id: str,
        target_stage: str
    ) -> StageResult:
        started_at = time.monotonic()
        logger.info(
            "deployment_recorded",
            extra={"asset_id": asset_id, "target_stage": target_stage}
        )
        return StageResult(
            stage="deploy",
            success=True,
            duration_seconds=time.monotonic() - started_at,
            output=f"recorded intent to deploy {asset_id} to {target_stage} (no-op placeholder)"
        )

    def _run_stage(
        self,
        stage: str,
        command: list[str]
    ) -> StageResult:
        started_at = time.monotonic()
        result = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True
        )
        output = (result.stdout or "")[-OUTPUT_TAIL_CHARS:] + (result.stderr or "")[-OUTPUT_TAIL_CHARS:]

        return StageResult(
            stage=stage,
            success=result.returncode == 0,
            duration_seconds=time.monotonic() - started_at,
            output=output
        )


class GitHubActionsRunTimeoutError(RuntimeError):
    pass


class GitHubActionsDeploymentPipeline:
    """
    DeploymentPipeline implementation for GitHub Actions, driven entirely
    through the gh CLI (workflow run / run list / run view) instead of a
    raw REST client with a token of its own - reuses whatever `gh auth
    login` session or GH_TOKEN/GITHUB_TOKEN env var is already available
    in the environment this runs in, so there's no HTTP/auth code here to
    get wrong. Same shell-out-to-existing-tooling approach as
    LocalDeploymentPipeline shelling out to pytest.

    Each of the four Protocol stages maps to an independently
    configurable workflow file (test_workflow/evaluation_workflow/
    experiment_workflow/deploy_workflow). A stage left as None is a soft
    no-op, same spirit as LocalDeploymentPipeline.deploy() being a
    placeholder - which of these stages actually needs a dedicated
    workflow is specific to a given repo's CI setup, not something to
    force by fabricating workflow files here.
    """
    name = "github_actions"

    def __init__(
        self,
        repo: str,
        test_workflow: str | None = None,
        evaluation_workflow: str | None = None,
        experiment_workflow: str | None = None,
        deploy_workflow: str | None = None,
        ref: str = "main",
        poll_interval_seconds: float = 5.0,
        poll_timeout_seconds: float = 600.0,
        gh_executable: str = "gh"
    ) -> None:
        self.repo = repo
        self.test_workflow = test_workflow
        self.evaluation_workflow = evaluation_workflow
        self.experiment_workflow = experiment_workflow
        self.deploy_workflow = deploy_workflow
        self.ref = ref
        self.poll_interval_seconds = poll_interval_seconds
        self.poll_timeout_seconds = poll_timeout_seconds
        self.gh_executable = gh_executable

    def run_tests(self) -> StageResult:
        return self._run_stage_via_workflow("test", self.test_workflow, {})

    def run_evaluation(self) -> StageResult:
        return self._run_stage_via_workflow("evaluate", self.evaluation_workflow, {})

    def run_experiment(
        self,
        experiment_id: str
    ) -> StageResult:
        return self._run_stage_via_workflow(
            "experiment",
            self.experiment_workflow,
            {"experiment_id": experiment_id}
        )

    def deploy(
        self,
        asset_id: str,
        target_stage: str
    ) -> StageResult:
        return self._run_stage_via_workflow(
            "deploy",
            self.deploy_workflow,
            {"asset_id": asset_id, "target_stage": target_stage}
        )

    def _run_stage_via_workflow(
        self,
        stage: str,
        workflow: str | None,
        inputs: dict[str, str]
    ) -> StageResult:
        started_at = time.monotonic()

        if workflow is None:
            return StageResult(
                stage=stage,
                success=True,
                duration_seconds=time.monotonic() - started_at,
                output=f"no workflow configured for stage '{stage}' - skipped"
            )

        run_ids_before = self._recent_run_ids(workflow)
        dispatch = self._gh(
            ["workflow", "run", workflow, "--repo", self.repo, "--ref", self.ref]
            + [arg for key, value in inputs.items() for arg in ("-f", f"{key}={value}")]
        )

        if dispatch.returncode != 0:
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=time.monotonic() - started_at,
                output=f"failed to dispatch workflow '{workflow}': {dispatch.stderr.strip()}"
            )

        try:
            run_id = self._wait_for_new_run(workflow, run_ids_before, started_at)
            conclusion = self._wait_for_completion(run_id, started_at)
        except GitHubActionsRunTimeoutError as ex:
            return StageResult(
                stage=stage,
                success=False,
                duration_seconds=time.monotonic() - started_at,
                output=str(ex)
            )

        return StageResult(
            stage=stage,
            success=conclusion == "success",
            duration_seconds=time.monotonic() - started_at,
            output=f"workflow '{workflow}' run {run_id} concluded: {conclusion}"
        )

    def _wait_for_new_run(
        self,
        workflow: str,
        run_ids_before: set[str],
        started_at: float
    ) -> str:
        while time.monotonic() - started_at < self.poll_timeout_seconds:
            new_ids = self._recent_run_ids(workflow) - run_ids_before

            if new_ids:
                return max(new_ids, key=int)

            time.sleep(self.poll_interval_seconds)

        raise GitHubActionsRunTimeoutError(
            f"timed out waiting for a new run of workflow '{workflow}' to appear"
        )

    def _wait_for_completion(
        self,
        run_id: str,
        started_at: float
    ) -> str:
        while time.monotonic() - started_at < self.poll_timeout_seconds:
            result = self._gh(
                ["run", "view", run_id, "--repo", self.repo, "--json", "status,conclusion"]
            )

            if result.returncode == 0:
                payload = json.loads(result.stdout)

                if payload.get("status") == "completed":
                    return payload.get("conclusion") or "unknown"

            time.sleep(self.poll_interval_seconds)

        raise GitHubActionsRunTimeoutError(
            f"timed out waiting for run {run_id} to complete"
        )

    def _recent_run_ids(
        self,
        workflow: str
    ) -> set[str]:
        result = self._gh(
            [
                "run", "list", "--repo", self.repo, "--workflow", workflow,
                "--limit", "20", "--json", "databaseId"
            ]
        )

        if result.returncode != 0:
            return set()

        return {str(entry["databaseId"]) for entry in json.loads(result.stdout)}

    def _gh(
        self,
        args: list[str]
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.gh_executable, *args],
            capture_output=True,
            text=True
        )
