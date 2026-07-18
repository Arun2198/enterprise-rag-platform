import logging
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from typing import Any

from mlops.schemas import ArtifactVersion

logger = logging.getLogger(__name__)


class ArtifactNotFoundError(KeyError):
    pass


class ArtifactRegistry:
    """
    Immutable, append-only version history for configuration-shaped
    artifacts: prompt templates, chunking configs, embedding configs,
    evaluation datasets, experiment definitions, policies, guardrail
    configs, feature definitions. `save()` never overwrites - it always
    appends a new version; nothing already saved is ever mutated.
    """

    def __init__(self) -> None:
        self._versions: dict[str, list[ArtifactVersion]] = {}

    def save(
        self,
        artifact_id: str,
        artifact_type: str,
        content: dict[str, Any],
        created_by: str | None = None
    ) -> ArtifactVersion:
        history = self._versions.setdefault(artifact_id, [])
        version = ArtifactVersion(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            version=len(history) + 1,
            content=content,
            created_at=_now_iso(),
            created_by=created_by
        )
        history.append(version)
        logger.info(
            "artifact_version_saved",
            extra={
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "version": version.version
            }
        )
        return version

    def get_latest(
        self,
        artifact_id: str
    ) -> ArtifactVersion:
        history = self._versions.get(artifact_id)

        if not history:
            raise ArtifactNotFoundError(artifact_id)

        return history[-1]

    def get_version(
        self,
        artifact_id: str,
        version: int
    ) -> ArtifactVersion:
        history = self._versions.get(artifact_id)

        if not history:
            raise ArtifactNotFoundError(artifact_id)

        matches = [entry for entry in history if entry.version == version]

        if not matches:
            raise ArtifactNotFoundError(f"{artifact_id} has no version {version}")

        return matches[0]

    def history(
        self,
        artifact_id: str
    ) -> list[ArtifactVersion]:
        if artifact_id not in self._versions:
            raise ArtifactNotFoundError(artifact_id)

        return list(self._versions[artifact_id])

    def list_artifact_ids(self) -> list[str]:
        return list(self._versions.keys())

    def export_state(self) -> dict[str, Any]:
        return {
            artifact_id: [asdict(version) for version in versions]
            for artifact_id, versions in self._versions.items()
        }

    def import_state(
        self,
        state: dict[str, Any]
    ) -> None:
        self._versions = {
            artifact_id: [ArtifactVersion(**raw) for raw in versions]
            for artifact_id, versions in state.items()
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
