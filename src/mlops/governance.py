import logging
from datetime import datetime
from datetime import timezone
from typing import Any

from mlops.schemas import ApprovalRecord
from mlops.schemas import AuditEvent
from mlops.schemas import LifecycleTransition

logger = logging.getLogger(__name__)


class PolicyViolationError(ValueError):
    pass


class GovernanceLog:
    """
    Audit trail and model lineage for the platform. Every approval,
    promotion, and policy check gets a durable AuditEvent. Lineage
    tracks which artifact versions (prompt template, chunking config,
    ...) were associated with a given model asset version, so "what
    actually produced this model in production" is answerable later.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lineage: dict[str, dict[str, int]] = {}
        self._counter = 0

    def record(
        self,
        actor: str | None,
        action: str,
        resource: str,
        details: dict[str, Any] | None = None
    ) -> AuditEvent:
        self._counter += 1
        event = AuditEvent(
            event_id=f"evt-{self._counter}",
            timestamp=_now_iso(),
            actor=actor,
            action=action,
            resource=resource,
            details=details or {}
        )
        self._events.append(event)
        logger.info(
            "audit_event",
            extra={"action": action, "resource": resource, "actor": actor}
        )
        return event

    def record_transition(
        self,
        transition: LifecycleTransition
    ) -> AuditEvent:
        return self.record(
            actor=transition.actor,
            action="lifecycle_transition",
            resource=transition.asset_id,
            details={
                "from_stage": transition.from_stage.value if transition.from_stage else None,
                "to_stage": transition.to_stage.value,
                "reason": transition.reason
            }
        )

    def record_approval(
        self,
        approval: ApprovalRecord
    ) -> AuditEvent:
        return self.record(
            actor=approval.approved_by,
            action="approval",
            resource=approval.asset_id,
            details={"to_stage": approval.to_stage.value, "comment": approval.comment}
        )

    def link_lineage(
        self,
        asset_id: str,
        artifact_id: str,
        artifact_version: int
    ) -> None:
        self._lineage.setdefault(asset_id, {})[artifact_id] = artifact_version
        self.record(
            actor=None,
            action="lineage_link",
            resource=asset_id,
            details={"artifact_id": artifact_id, "artifact_version": artifact_version}
        )

    def lineage(
        self,
        asset_id: str
    ) -> dict[str, int]:
        return dict(self._lineage.get(asset_id, {}))

    def history(
        self,
        resource: str | None = None
    ) -> list[AuditEvent]:
        if resource is None:
            return list(self._events)

        return [event for event in self._events if event.resource == resource]

    def check_policy(
        self,
        rule_name: str,
        condition: bool,
        message: str
    ) -> None:
        """
        Simple assertion-style compliance check: records the outcome
        either way (so a passed check is just as visible in the audit
        trail as a failed one), then raises if the condition failed.
        """
        self.record(
            actor=None,
            action="policy_check",
            resource=rule_name,
            details={"passed": condition, "message": message}
        )

        if not condition:
            raise PolicyViolationError(f"{rule_name}: {message}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
