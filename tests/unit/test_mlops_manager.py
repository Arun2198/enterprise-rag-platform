import pytest

from mlops.backup import BackupManager
from mlops.deployment import LocalDeploymentPipeline
from mlops.lifecycle import InvalidTransitionError
from mlops.manager import PlatformManager
from mlops.manager import ProviderNotFoundError
from mlops.schemas import AssetType
from mlops.schemas import LifecycleStage


def test_register_provider_and_get_provider():

    manager = PlatformManager()
    pipeline = LocalDeploymentPipeline()

    manager.register_provider("ci", pipeline)

    assert manager.get_provider("ci") is pipeline


def test_get_unknown_provider_raises():

    manager = PlatformManager()

    with pytest.raises(ProviderNotFoundError):
        manager.get_provider("does-not-exist")


def test_register_asset_records_a_governance_event():

    manager = PlatformManager()

    asset = manager.register_asset(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    events = manager.governance.history(resource=asset.asset_id)
    assert any(event.action == "register_asset" for event in events)


def test_promote_records_transition_and_approval_in_governance():

    manager = PlatformManager()
    asset = manager.register_asset(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    manager.promote(asset.asset_id, LifecycleStage.VALIDATION, actor="alice")
    manager.promote(asset.asset_id, LifecycleStage.STAGING, approved_by="bob")

    events = manager.governance.history(resource=asset.asset_id)
    actions = [event.action for event in events]
    assert "lifecycle_transition" in actions
    assert "approval" in actions
    assert manager.registry.get(asset.asset_id).status == LifecycleStage.STAGING


def test_promote_failure_does_not_record_governance_event():

    manager = PlatformManager()
    asset = manager.register_asset(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    events_before = len(manager.governance.history())

    with pytest.raises(InvalidTransitionError):
        manager.promote(asset.asset_id, LifecycleStage.PRODUCTION, approved_by="bob")

    assert len(manager.governance.history()) == events_before


def test_backup_and_restore_round_trip_through_manager(tmp_path):

    manager = PlatformManager(backup=BackupManager(output_dir=str(tmp_path)))
    manager.register_asset(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    snapshot = manager.create_backup()

    restored_manager = PlatformManager()
    restored = restored_manager.restore_backup(snapshot.path)

    assert "registry" in restored
    assert restored_manager.registry.list() == manager.registry.list()


def test_link_asset_lineage_delegates_to_governance():

    manager = PlatformManager()
    asset = manager.register_asset(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    artifact = manager.artifacts.save("chunking-default", "chunking_config", {"chunk_size": 900})

    manager.link_asset_lineage(asset.asset_id, artifact.artifact_id, artifact.version)

    assert manager.governance.lineage(asset.asset_id) == {"chunking-default": 1}


def test_registry_to_history_end_to_end(tmp_path):
    """
    Registry -> Lifecycle -> Deployment -> Observability -> Governance ->
    History, end-to-end through PlatformManager.
    """
    manager = PlatformManager()

    # Registry: register a new embedding model version
    asset = manager.register_asset(
        AssetType.EMBEDDING_MODEL,
        "hashing",
        "2.0",
        metadata={"dimensions": 384}
    )
    assert asset.status == LifecycleStage.DEVELOPMENT

    # Deployment: a CI pipeline is registered and used to validate the change
    pipeline = LocalDeploymentPipeline(repo_root=tmp_path)
    manager.register_provider("ci", pipeline)
    deploy_result = manager.get_provider("ci").deploy(asset.asset_id, "staging")
    assert deploy_result.success is True

    # Lifecycle: promote through validation into staging (approval required)
    manager.promote(asset.asset_id, LifecycleStage.VALIDATION, actor="alice")
    manager.promote(
        asset.asset_id,
        LifecycleStage.STAGING,
        approved_by="bob",
        reason="passed CI"
    )
    assert manager.registry.get(asset.asset_id).status == LifecycleStage.STAGING

    # Observability: registering + promoting both emit OpenTelemetry
    # metrics (see test_mlops_telemetry.py for direct assertions on this;
    # here we just confirm the calls didn't raise, which they wouldn't
    # even without a configured exporter)

    # Governance: every step left an audit trail entry for this asset
    events = manager.governance.history(resource=asset.asset_id)
    actions = [event.action for event in events]
    assert "register_asset" in actions
    assert "lifecycle_transition" in actions
    assert "approval" in actions

    # History: lifecycle + governance histories agree on what happened
    lifecycle_history = manager.lifecycle.history(asset.asset_id)
    assert [t.to_stage for t in lifecycle_history] == [
        LifecycleStage.VALIDATION,
        LifecycleStage.STAGING
    ]
