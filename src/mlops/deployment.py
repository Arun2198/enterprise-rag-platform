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
    driven by GitHub Actions, Azure DevOps, Jenkins, GitLab CI, or
    in-process (LocalDeploymentPipeline below) - a CI system's job is to
    call these stages in order and act on the result. None of this is
    provider-specific; a real GitHub Actions/Azure DevOps/Jenkins/
    GitLab CI adapter is not implemented here (that's inherently a
    matter of calling that provider's own REST API/CLI), but it would
    implement this exact Protocol so PlatformManager and everything
    downstream is unaffected by which CI system actually runs it.
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
