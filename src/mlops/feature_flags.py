import hashlib
import logging
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from typing import Any

from mlops.schemas import FeatureFlag

logger = logging.getLogger(__name__)


class FlagNotFoundError(KeyError):
    pass


class FeatureFlagManager:
    """
    Runtime feature toggles - enable/disable the reranker or guardrails,
    switch embedding model or LLM provider, percentage-based canary
    rollout, and shadow deployment (marks a flag as "evaluate but don't
    let it affect the returned result" - what shadow mode actually does
    is up to the caller's own code; this just tracks the flag's state).
    """

    def __init__(self) -> None:
        self._flags: dict[str, FeatureFlag] = {}

    def define(
        self,
        name: str,
        enabled: bool = False,
        rollout_percentage: float = 100.0,
        shadow: bool = False,
        description: str | None = None
    ) -> FeatureFlag:
        self._validate_percentage(rollout_percentage)
        flag = FeatureFlag(
            name=name,
            enabled=enabled,
            rollout_percentage=rollout_percentage,
            shadow=shadow,
            description=description,
            updated_at=_now_iso()
        )
        self._flags[name] = flag
        logger.info("feature_flag_defined", extra={"flag": name, "enabled": enabled})
        return flag

    def set_enabled(
        self,
        name: str,
        enabled: bool
    ) -> FeatureFlag:
        return self._update(name, enabled=enabled)

    def set_rollout_percentage(
        self,
        name: str,
        percentage: float
    ) -> FeatureFlag:
        self._validate_percentage(percentage)
        return self._update(name, rollout_percentage=percentage)

    def set_shadow(
        self,
        name: str,
        shadow: bool
    ) -> FeatureFlag:
        return self._update(name, shadow=shadow)

    def get(
        self,
        name: str
    ) -> FeatureFlag:
        if name not in self._flags:
            raise FlagNotFoundError(name)

        return self._flags[name]

    def list(self) -> list[FeatureFlag]:
        return list(self._flags.values())

    def is_enabled_for(
        self,
        name: str,
        subject_id: str
    ) -> bool:
        """
        Deterministic canary check: the same subject_id always gets the
        same answer for a given flag + rollout_percentage (stable hash
        bucketing), so a user doesn't flip in and out of a rollout
        between requests.
        """
        flag = self.get(name)

        if not flag.enabled:
            return False

        if flag.rollout_percentage >= 100.0:
            return True

        if flag.rollout_percentage <= 0.0:
            return False

        bucket = int(hashlib.sha256(f"{name}:{subject_id}".encode()).hexdigest(), 16) % 100
        return bucket < flag.rollout_percentage

    def export_state(self) -> dict[str, Any]:
        return {"flags": [asdict(flag) for flag in self._flags.values()]}

    def import_state(
        self,
        state: dict[str, Any]
    ) -> None:
        self._flags = {}

        for raw in state.get("flags", []):
            flag = FeatureFlag(**raw)
            self._flags[flag.name] = flag

    def _update(
        self,
        name: str,
        **changes: Any
    ) -> FeatureFlag:
        current = self.get(name)
        updated = replace(current, updated_at=_now_iso(), **changes)
        self._flags[name] = updated
        logger.info("feature_flag_updated", extra={"flag": name, **changes})
        return updated

    def _validate_percentage(
        self,
        percentage: float
    ) -> None:
        if not 0.0 <= percentage <= 100.0:
            raise ValueError("rollout_percentage must be between 0 and 100")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
