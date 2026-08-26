import pytest

from mlops.artifacts import ArtifactRegistry
from mlops.backup import BackupManager
from mlops.backup import S3BackupTarget
from mlops.recovery import RecoveryManager
from mlops.recovery import SnapshotNotFoundError
from mlops.registry import ModelRegistry
from mlops.schemas import AssetType


class FakePaginator:

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def paginate(self, Bucket, Prefix):
        keys = [key for key in self._objects if key.startswith(Prefix)]
        yield {"Contents": [{"Key": key} for key in keys]}


class FakeBody:

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class FakeS3Client:

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": FakeBody(self.objects[Key])}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self.objects)


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


def test_create_snapshot_uploads_to_target_when_configured(tmp_path):

    client = FakeS3Client()
    target = S3BackupTarget(client=client, bucket_name="my-bucket")
    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    manager = BackupManager(output_dir=str(tmp_path), target=target)

    snapshot = manager.create_snapshot({"registry": registry})

    uploaded = target.download(snapshot.snapshot_id)
    assert uploaded["registry"] == registry.export_state()


def test_create_snapshot_does_not_touch_target_when_none_configured(tmp_path):

    registry = ModelRegistry()
    manager = BackupManager(output_dir=str(tmp_path))

    # no target configured - should not raise, no S3 interaction at all
    manager.create_snapshot({"registry": registry})


def test_s3_backup_target_list_snapshot_ids_returns_uploaded_ids(tmp_path):

    client = FakeS3Client()
    target = S3BackupTarget(client=client, bucket_name="my-bucket")
    registry = ModelRegistry()
    manager = BackupManager(output_dir=str(tmp_path), target=target)

    snapshot = manager.create_snapshot({"registry": registry})

    assert target.list_snapshot_ids() == [snapshot.snapshot_id]


def test_restore_from_target_repopulates_fresh_component(tmp_path):

    client = FakeS3Client()
    target = S3BackupTarget(client=client, bucket_name="my-bucket")
    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    manager = BackupManager(output_dir=str(tmp_path), target=target)
    snapshot = manager.create_snapshot({"registry": registry})

    fresh_registry = ModelRegistry()
    recovery_manager = RecoveryManager()
    restored = recovery_manager.restore_from_target(
        snapshot.snapshot_id, target, {"registry": fresh_registry}
    )

    assert restored == ["registry"]
    assert fresh_registry.list() == registry.list()
