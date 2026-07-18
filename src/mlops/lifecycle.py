import logging
from datetime import datetime
from datetime import timezone

from mlops.registry import ModelRegistry
from mlops.schemas import ApprovalRecord
from mlops.schemas import LifecycleStage
from mlops.schemas import LifecycleTransition
from mlops.schemas import ModelAsset

logger = logging.getLogger(__name__)

ALLOWED_TRANSITIONS: dict[LifecycleStage, frozenset[LifecycleStage]] = {
    LifecycleStage.DEVELOPMENT: frozenset({LifecycleStage.VALIDATION, LifecycleStage.RETIRED}),
    LifecycleStage.VALIDATION: frozenset({
        LifecycleStage.STAGING, LifecycleStage.DEVELOPMENT, LifecycleStage.RETIRED
    }),
    LifecycleStage.STAGING: frozenset({
        LifecycleStage.PRODUCTION, LifecycleStage.VALIDATION, LifecycleStage.RETIRED
    }),
    LifecycleStage.PRODUCTION: frozenset({LifecycleStage.RETIRED}),
    LifecycleStage.RETIRED: frozenset(),
}

STAGES_REQUIRING_APPROVAL = frozenset({LifecycleStage.STAGING, LifecycleStage.PRODUCTION})


class InvalidTransitionError(ValueError):
    pass


class ApprovalRequiredError(ValueError):
    pass


class LifecycleManager:
    """
    Promotion state machine: Development -> Validation -> Staging ->
    Production -> Retired, plus the reject-back-a-stage edges (Validation
    can bounce back to Development, Staging back to Validation). Moving
    into Staging or Production requires an `approved_by` - everything
    else can be self-service. Every transition and approval is recorded
    here; PlatformManager additionally mirrors both into GovernanceLog.
    """

    def __init__(
        self,
        registry: ModelRegistry
    ) -> None:
        self.registry = registry
        self._transitions: list[LifecycleTransition] = []
        self._approvals: list[ApprovalRecord] = []

    def promote(
        self,
        asset_id: str,
        to_stage: LifecycleStage,
        actor: str | None = None,
        reason: str | None = None,
        approved_by: str | None = None
    ) -> ModelAsset:
        asset = self.registry.get(asset_id)
        current_stage = asset.status
        allowed = ALLOWED_TRANSITIONS.get(current_stage, frozenset())

        if to_stage not in allowed:
            raise InvalidTransitionError(
                f"cannot transition {asset_id} from {current_stage.value} to {to_stage.value}"
            )

        if to_stage in STAGES_REQUIRING_APPROVAL and approved_by is None:
            raise ApprovalRequiredError(
                f"promoting {asset_id} to {to_stage.value} requires approved_by"
            )

        updated = self.registry.update_status(asset_id, to_stage)

        self._transitions.append(
            LifecycleTransition(
                asset_id=asset_id,
                from_stage=current_stage,
                to_stage=to_stage,
                timestamp=_now_iso(),
                actor=actor,
                reason=reason
            )
        )

        if approved_by is not None:
            self._approvals.append(
                ApprovalRecord(
                    asset_id=asset_id,
                    to_stage=to_stage,
                    approved_by=approved_by,
                    timestamp=_now_iso(),
                    comment=reason
                )
            )

        logger.info(
            "lifecycle_transition",
            extra={
                "asset_id": asset_id,
                "from_stage": current_stage.value,
                "to_stage": to_stage.value,
                "actor": actor
            }
        )
        return updated

    def history(
        self,
        asset_id: str
    ) -> list[LifecycleTransition]:
        return [t for t in self._transitions if t.asset_id == asset_id]

    def approvals(
        self,
        asset_id: str
    ) -> list[ApprovalRecord]:
        return [a for a in self._approvals if a.asset_id == asset_id]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
