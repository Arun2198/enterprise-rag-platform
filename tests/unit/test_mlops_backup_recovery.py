import pytest

from mlops.artifacts import ArtifactRegistry
from mlops.backup import BackupManager
from mlops.recovery import RecoveryManager
from mlops.recovery import SnapshotNotFoundError
from mlops.registry import ModelRegistry
from mlops.schemas import AssetType


def test_create_snapshot_writes_a_file_with_expected_components(tmp_path):

    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    manager = BackupManager(output_dir=str(tmp_path))

    snapshot = manager.create_snapshot({"registry": registry})

    assert snapshot.components == ["registry"]
    assert (tmp_path / f"{snapshot.snapshot_id}.json").exists()


def test_list_snapshots_returns_written_files(tmp_path):

    registry = ModelRegistry()
    manager = BackupManager(output_dir=str(tmp_path))
    manager.create_snapshot({"registry": registry})

    assert len(manager.list_snapshots()) == 1


def test_list_snapshots_empty_when_no_backups_yet(tmp_path):

    manager = BackupManager(output_dir=str(tmp_path / "does-not-exist-yet"))

    assert manager.list_snapshots() == []


def test_restore_snapshot_repopulates_fresh_component(tmp_path):

    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    backup_manager = BackupManager(output_dir=str(tmp_path))
    snapshot = backup_manager.create_snapshot({"registry": registry})

    fresh_registry = ModelRegistry()
    recovery_manager = RecoveryManager()
    restored = recovery_manager.restore_snapshot(snapshot.path, {"registry": fresh_registry})

    assert restored == ["registry"]
    assert fresh_registry.list() == registry.list()


def test_restore_skips_components_not_present_in_snapshot(tmp_path):

    registry = ModelRegistry()
    backup_manager = BackupManager(output_dir=str(tmp_path))
    snapshot = backup_manager.create_snapshot({"registry": registry})

    recovery_manager = RecoveryManager()
    restored = recovery_manager.restore_snapshot(
        snapshot.path,
        {"registry": ModelRegistry(), "artifacts": ArtifactRegistry()}
    )

    assert restored == ["registry"]


def test_restore_missing_snapshot_raises(tmp_path):

    recovery_manager = RecoveryManager()

    with pytest.raises(SnapshotNotFoundError):
        recovery_manager.restore_snapshot(str(tmp_path / "does-not-exist.json"), {})


def test_inspect_snapshot_lists_components(tmp_path):

    registry = ModelRegistry()
    artifacts = ArtifactRegistry()
    backup_manager = BackupManager(output_dir=str(tmp_path))
    snapshot = backup_manager.create_snapshot({"registry": registry, "artifacts": artifacts})

    recovery_manager = RecoveryManager()
    info = recovery_manager.inspect_snapshot(snapshot.path)

    assert set(info["components"]) == {"registry", "artifacts"}
