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
    A durable cloud backup destination (S3, Azure Blob Storage, GCS).
    BackupManager below always writes snapshots to the local filesystem
    first; a BackupTarget additionally uploads the same JSON payload
    somewhere that survives the local disk disappearing - which it does
    on every ECS Fargate task restart/redeploy, so local-only backup was
    never actually durable for the live app. S3BackupTarget is the one
    real implementation; Azure Blob/GCS are still extension points.
    """

    def upload(self, snapshot: BackupSnapshot, payload: dict[str, Any]) -> None:
        ...

    def download(self, snapshot_id: str) -> dict[str, Any]:
        ...

    def list_snapshot_ids(self) -> list[str]:
        ...


class S3BackupTarget:
    """
    Real BackupTarget backed by S3 - same "S3 instead of a new
    database/service" pattern as ingestion_job_store.IngestionJobStore
    and ingestion.s3_document_store.S3DocumentStore: one small JSON
    object per snapshot (`{prefix}{snapshot_id}.json`), no separate
    infra to provision.
    """

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        prefix: str = "mlops_backups/"
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.prefix = prefix

    def upload(
        self,
        snapshot: BackupSnapshot,
        payload: dict[str, Any]
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=self._key(snapshot.snapshot_id),
            Body=json.dumps(payload).encode("utf-8"),
            ContentType="application/json"
        )

    def download(
        self,
        snapshot_id: str
    ) -> dict[str, Any]:
        response = self.client.get_object(Bucket=self.bucket_name, Key=self._key(snapshot_id))
        return json.loads(response["Body"].read())

    def list_snapshot_ids(self) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        ids = []

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]

                if key.endswith(".json"):
                    ids.append(Path(key).stem)

        return sorted(ids)

    def _key(
        self,
        snapshot_id: str
    ) -> str:
        return f"{self.prefix}{snapshot_id}.json"


class BackupManager:
    """
    Serializes the current state of one or more platform components -
    anything satisfying ExportableComponent (an `.export_state() ->
    dict` method) - to a timestamped local JSON snapshot. Configuration,
    artifact, and registry backup are all just "pass the right
    component in". When a `target` (a BackupTarget, e.g. S3BackupTarget)
    is supplied, every snapshot is also uploaded there right after the
    local write - the local file stays a fast working copy, the target
    is the durable source of truth. `target=None` (the default) keeps
    behavior exactly as before this existed - local-only, no AWS call.
    """

    def __init__(
        self,
        output_dir: str = "mlops_backups",
        target: BackupTarget | None = None
    ) -> None:
        self.output_dir = Path(output_dir)
        self.target = target

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

        if self.target is not None:
            self.target.upload(snapshot, payload)
            logger.info(
                "backup_snapshot_uploaded",
                extra={"snapshot_id": snapshot_id, "target": type(self.target).__name__}
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
