import pytest

from mlops.registry import AssetNotFoundError
from mlops.registry import DuplicateAssetError
from mlops.registry import ModelRegistry
from mlops.schemas import AssetType
from mlops.schemas import LifecycleStage


def test_register_creates_asset_in_development_by_default():

    registry = ModelRegistry()

    asset = registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    assert asset.status == LifecycleStage.DEVELOPMENT
    assert asset.asset_id == "embedding_model:hashing:1.0"


def test_register_duplicate_raises():

    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    with pytest.raises(DuplicateAssetError):
        registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")


def test_get_missing_asset_raises():

    registry = ModelRegistry()

    with pytest.raises(AssetNotFoundError):
        registry.get("does-not-exist")


def test_list_filters_by_asset_type_and_status():

    registry = ModelRegistry()
    registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")
    registry.register(AssetType.RERANKER, "ms-marco", "1.0")
    prod_asset = registry.register(AssetType.EMBEDDING_MODEL, "bge", "1.0")
    registry.update_status(prod_asset.asset_id, LifecycleStage.PRODUCTION)

    embedding_assets = registry.list(asset_type=AssetType.EMBEDDING_MODEL)
    production_assets = registry.list(status=LifecycleStage.PRODUCTION)

    assert len(embedding_assets) == 2
    assert len(production_assets) == 1
    assert production_assets[0].asset_id == prod_asset.asset_id


def test_update_status_sets_updated_at():

    registry = ModelRegistry()
    asset = registry.register(AssetType.EMBEDDING_MODEL, "hashing", "1.0")

    updated = registry.update_status(asset.asset_id, LifecycleStage.VALIDATION)

    assert updated.status == LifecycleStage.VALIDATION
    assert updated.updated_at is not None


def test_export_import_state_round_trips():

    registry = ModelRegistry()
    registry.register(
        AssetType.LLM_PROVIDER,
        "openai-compatible",
        "1.0",
        metadata={"base_url": "https://example.com"}
    )

    state = registry.export_state()
    restored = ModelRegistry()
    restored.import_state(state)

    assert restored.list() == registry.list()
