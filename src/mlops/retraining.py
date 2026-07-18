from typing import Protocol

from mlops.schemas import RetrainingRequest

RETRAINING_TRIGGERS = ("scheduled", "drift", "manual")


class RetrainingTrigger(Protocol):
    """
    Not implemented - no model training happens in this repo. This is
    the extension point a real MLOps pipeline would implement:
    scheduled (via Scheduler), drift-triggered (via a DriftDetector
    crossing its threshold), or manual (an operator call) all end up
    producing a RetrainingRequest through request(); what a concrete
    trigger actually does to kick off training is out of scope here.
    """

    def request(
        self,
        asset_id: str,
        trigger: str,
        reason: str | None = None
    ) -> RetrainingRequest:
        ...


class ValidationWorkflow(Protocol):
    """
    Not implemented - the gate a retrained model would need to pass
    (e.g. an evaluation.runner.EvaluationRunner run against a regression
    threshold via evaluation.report.compare_reports) before promotion
    through LifecycleManager.
    """

    def validate(
        self,
        asset_id: str
    ) -> bool:
        ...
