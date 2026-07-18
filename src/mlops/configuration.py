import logging
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Callable

from mlops.schemas import ConfigProfile

logger = logging.getLogger(__name__)


class ProfileNotFoundError(KeyError):
    pass


class ConfigValidationError(ValueError):
    pass


class NoActiveProfileError(ValueError):
    pass


class NoRollbackTargetError(ValueError):
    pass


class ConfigurationManager:
    """
    Named environment profiles (dev/staging/prod/...), each with its own
    versioned, append-only history - saving a profile never overwrites a
    previous version. Runtime overrides layer on top of whichever
    profile version is active without mutating anything stored.
    `validators` maps a config key to a callable that raises on an
    invalid value; failing validation rejects the save entirely (nothing
    partially-invalid ever enters history).
    """

    def __init__(
        self,
        validators: dict[str, Callable[[Any], None]] | None = None
    ) -> None:
        self._profiles: dict[str, list[ConfigProfile]] = {}
        self._active_profile: str | None = None
        self._active_version: int | None = None
        self._overrides: dict[str, Any] = {}
        self._validators = validators or {}

    def save_profile(
        self,
        name: str,
        values: dict[str, Any],
        description: str | None = None
    ) -> ConfigProfile:
        self._validate(values)
        history = self._profiles.setdefault(name, [])
        profile = ConfigProfile(
            name=name,
            values=dict(values),
            version=len(history) + 1,
            created_at=_now_iso(),
            description=description
        )
        history.append(profile)
        logger.info(
            "config_profile_saved",
            extra={"profile": name, "version": profile.version}
        )
        return profile

    def activate(
        self,
        name: str,
        version: int | None = None
    ) -> ConfigProfile:
        profile = self._get_version(name, version)
        self._active_profile = name
        self._active_version = profile.version
        logger.info(
            "config_profile_activated",
            extra={"profile": name, "version": profile.version}
        )
        return profile

    def rollback(
        self,
        name: str | None = None
    ) -> ConfigProfile:
        target_name = name or self._active_profile

        if target_name is None:
            raise NoActiveProfileError("no active profile to roll back")

        history = self._profiles.get(target_name)

        if not history:
            raise ProfileNotFoundError(target_name)

        current_version = (
            self._active_version
            if self._active_profile == target_name
            else history[-1].version
        )
        target_version = current_version - 1

        if target_version < 1:
            raise NoRollbackTargetError(f"no earlier version to roll back to for {target_name}")

        return self.activate(target_name, version=target_version)

    def active_profile(self) -> ConfigProfile:
        if self._active_profile is None:
            raise NoActiveProfileError("no active profile")

        return self._get_version(self._active_profile, self._active_version)

    def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        if key in self._overrides:
            return self._overrides[key]

        if self._active_profile is None:
            return default

        return self.active_profile().values.get(key, default)

    def set_override(
        self,
        key: str,
        value: Any
    ) -> None:
        self._overrides[key] = value
        logger.info("config_override_set", extra={"key": key})

    def clear_override(
        self,
        key: str
    ) -> None:
        self._overrides.pop(key, None)

    def history(
        self,
        name: str
    ) -> list[ConfigProfile]:
        if name not in self._profiles:
            raise ProfileNotFoundError(name)

        return list(self._profiles[name])

    def export_state(self) -> dict[str, Any]:
        return {
            "profiles": {
                name: [asdict(profile) for profile in versions]
                for name, versions in self._profiles.items()
            },
            "active_profile": self._active_profile,
            "active_version": self._active_version,
            "overrides": dict(self._overrides)
        }

    def import_state(
        self,
        state: dict[str, Any]
    ) -> None:
        self._profiles = {
            name: [ConfigProfile(**raw) for raw in versions]
            for name, versions in state.get("profiles", {}).items()
        }
        self._active_profile = state.get("active_profile")
        self._active_version = state.get("active_version")
        self._overrides = dict(state.get("overrides", {}))

    def _get_version(
        self,
        name: str,
        version: int | None
    ) -> ConfigProfile:
        history = self._profiles.get(name)

        if not history:
            raise ProfileNotFoundError(name)

        if version is None:
            return history[-1]

        matches = [profile for profile in history if profile.version == version]

        if not matches:
            raise ProfileNotFoundError(f"{name} has no version {version}")

        return matches[0]

    def _validate(
        self,
        values: dict[str, Any]
    ) -> None:
        for key, validator in self._validators.items():
            if key not in values:
                continue

            try:
                validator(values[key])
            except Exception as ex:
                raise ConfigValidationError(f"validation failed for '{key}': {ex}") from ex


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
