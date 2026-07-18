import json
import logging
from pathlib import Path
from typing import Any

from mlops.backup import ExportableComponent

logger = logging.getLogger(__name__)


class SnapshotNotFoundError(FileNotFoundError):
    pass


class RecoveryManager:
    """
    Restores component state from a local JSON snapshot written by
    BackupManager. Only restores components the caller explicitly passes
    in via the same `.import_state(data)` contract every backed-up
    component implements - never silently restores something the caller
    didn't ask for, and silently skips any component present in the
    snapshot but not requested.
    """

    def restore_snapshot(
        self,
        path: str,
        components: dict[str, ExportableComponent]
    ) -> list[str]:
        snapshot_path = Path(path)

        if not snapshot_path.exists():
            raise SnapshotNotFoundError(path)

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        restored = []

        for name, component in components.items():
            if name not in payload:
                continue

            component.import_state(payload[name])
            restored.append(name)

        logger.info(
            "snapshot_restored",
            extra={"path": path, "components": restored}
        )
        return restored

    def inspect_snapshot(
        self,
        path: str
    ) -> dict[str, Any]:
        snapshot_path = Path(path)

        if not snapshot_path.exists():
            raise SnapshotNotFoundError(path)

        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return {"path": path, "components": list(payload.keys())}
