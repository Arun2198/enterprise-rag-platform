import pytest

from mlops.lifecycle import ApprovalRequiredError
from mlops.lifecycle import InvalidTransitionError
from mlops.lifecycle import LifecycleManager
from mlops.registry import ModelRegistry
from mlops.schemas import AssetType
from mlops.schemas import LifecycleStage


def _new_manager():
    registry = ModelRegistry()
    asset = registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    return LifecycleManager(registry), asset.asset_id


def test_development_to_validation_needs_no_approval():

    manager, asset_id = _new_manager()

    updated = manager.promote(asset_id, LifecycleStage.VALIDATION, actor="alice")

    assert updated.status == LifecycleStage.VALIDATION


def test_promoting_to_staging_without_approval_raises():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.VALIDATION)

    with pytest.raises(ApprovalRequiredError):
        manager.promote(asset_id, LifecycleStage.STAGING)


def test_promoting_to_staging_with_approval_succeeds():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.VALIDATION)

    updated = manager.promote(asset_id, LifecycleStage.STAGING, approved_by="bob")

    assert updated.status == LifecycleStage.STAGING
    assert manager.approvals(asset_id)[0].approved_by == "bob"


def test_illegal_transition_raises():

    manager, asset_id = _new_manager()

    with pytest.raises(InvalidTransitionError):
        manager.promote(asset_id, LifecycleStage.PRODUCTION, approved_by="bob")


def test_full_promotion_path_to_production():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.VALIDATION)
    manager.promote(asset_id, LifecycleStage.STAGING, approved_by="bob")
    updated = manager.promote(asset_id, LifecycleStage.PRODUCTION, approved_by="carol")

    assert updated.status == LifecycleStage.PRODUCTION


def test_retired_asset_cannot_transition_further():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.RETIRED)

    with pytest.raises(InvalidTransitionError):
        manager.promote(asset_id, LifecycleStage.VALIDATION)


def test_validation_can_reject_back_to_development():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.VALIDATION)

    updated = manager.promote(asset_id, LifecycleStage.DEVELOPMENT, reason="failed checks")

    assert updated.status == LifecycleStage.DEVELOPMENT


def test_history_tracks_every_transition_in_order():

    manager, asset_id = _new_manager()
    manager.promote(asset_id, LifecycleStage.VALIDATION)
    manager.promote(asset_id, LifecycleStage.STAGING, approved_by="bob")

    history = manager.history(asset_id)

    assert [t.to_stage for t in history] == [LifecycleStage.VALIDATION, LifecycleStage.STAGING]
    assert history[0].from_stage == LifecycleStage.DEVELOPMENT
