import json
import logging
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Protocol

from mlops.schemas import BackupSnapshot

logger = logging.getLogger(__name__)


class ExportableComponent(Protocol):
    """Any component BackupManager/RecoveryManager can snapshot - registry.ModelRegistry,
    artifacts.ArtifactRegistry, configuration.ConfigurationManager, and
    feature_flags.FeatureFlagManager all implement this shape."""

    def export_state(self) -> dict[str, Any]:
        ...

    def import_state(self, state: dict[str, Any]) -> None:
        ...


class BackupTarget(Protocol):
    """
    Extension point (not implemented) for a cloud backup destination
    (S3, Azure Blob Storage, GCS). BackupManager below writes snapshots
    to the local filesystem directly; a concrete BackupTarget would
    additionally upload the same JSON payload somewhere durable.
    """

    def upload(self, snapshot: BackupSnapshot, payload: dict[str, Any]) -> None:
        ...


class BackupManager:
    """
    Serializes the current state of one or more platform components -
    anything satisfying ExportableComponent (an `.export_state() ->
    dict` method) - to a timestamped local JSON snapshot. Configuration,
    artifact, and registry backup are all just "pass the right
    component in".
    """

    def __init__(
        self,
        output_dir: str = "mlops_backups"
    ) -> None:
        self.output_dir = Path(output_dir)

    def create_snapshot(
        self,
        components: dict[str, ExportableComponent]
    ) -> BackupSnapshot:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"snapshot_{timestamp}"
        path = self.output_dir / f"{snapshot_id}.json"

        payload = {
            name: component.export_state()
            for name, component in components.items()
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        snapshot = BackupSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            components=list(components.keys()),
            path=str(path)
        )
        logger.info(
            "backup_snapshot_created",
            extra={"snapshot_id": snapshot_id, "components": snapshot.components}
        )
        return snapshot

    def list_snapshots(self) -> list[str]:
        if not self.output_dir.exists():
            return []

        return sorted(str(path) for path in self.output_dir.glob("snapshot_*.json"))
