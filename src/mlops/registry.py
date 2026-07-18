import logging
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Protocol

from mlops.schemas import AssetType
from mlops.schemas import LifecycleStage
from mlops.schemas import ModelAsset

logger = logging.getLogger(__name__)


class AssetNotFoundError(KeyError):
    pass


class DuplicateAssetError(ValueError):
    pass


class ModelRegistryBackend(Protocol):
    """
    Extension point for a real MLflow Model Registry or a cloud
    provider's registry (Azure ML, SageMaker, Vertex AI). Not
    implemented - ModelRegistry below is the local, provider-agnostic
    reference implementation; a concrete backend satisfies this same
    shape and can be swapped in without changing any caller.
    """

    def register(self, asset: ModelAsset) -> None:
        ...

    def get(self, asset_id: str) -> ModelAsset:
        ...

    def list(self, asset_type: AssetType | None = None) -> list[ModelAsset]:
        ...

    def update_status(self, asset_id: str, status: LifecycleStage) -> ModelAsset:
        ...


class ModelRegistry:
    """
    Local, in-memory registry of versioned AI assets - embedding models,
    rerankers, LLM providers, prompt templates, guardrail models,
    evaluation models. Tracks version, status, and metadata per asset.
    """

    def __init__(self) -> None:
        self._assets: dict[str, ModelAsset] = {}

    def register(
        self,
        asset_type: AssetType,
        name: str,
        version: str,
        metadata: dict[str, Any] | None = None,
        status: LifecycleStage = LifecycleStage.DEVELOPMENT
    ) -> ModelAsset:
        asset_id = self._asset_id(asset_type, name, version)

        if asset_id in self._assets:
            raise DuplicateAssetError(f"asset already registered: {asset_id}")

        asset = ModelAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            name=name,
            version=version,
            status=status,
            metadata=metadata or {},
            registered_at=_now_iso()
        )
        self._assets[asset_id] = asset
        logger.info(
            "model_asset_registered",
            extra={"asset_id": asset_id, "asset_type": asset_type.value, "status": status.value}
        )
        return asset

    def get(
        self,
        asset_id: str
    ) -> ModelAsset:
        if asset_id not in self._assets:
            raise AssetNotFoundError(asset_id)

        return self._assets[asset_id]

    def list(
        self,
        asset_type: AssetType | None = None,
        status: LifecycleStage | None = None
    ) -> list[ModelAsset]:
        assets = list(self._assets.values())

        if asset_type is not None:
            assets = [asset for asset in assets if asset.asset_type == asset_type]

        if status is not None:
            assets = [asset for asset in assets if asset.status == status]

        return assets

    def update_status(
        self,
        asset_id: str,
        status: LifecycleStage
    ) -> ModelAsset:
        current = self.get(asset_id)
        updated = replace(current, status=status, updated_at=_now_iso())
        self._assets[asset_id] = updated
        logger.info(
            "model_asset_status_updated",
            extra={"asset_id": asset_id, "status": status.value}
        )
        return updated

    def export_state(self) -> dict[str, Any]:
        return {"assets": [asdict(asset) for asset in self._assets.values()]}

    def import_state(
        self,
        state: dict[str, Any]
    ) -> None:
        self._assets = {}

        for raw in state.get("assets", []):
            asset = ModelAsset(
                asset_id=raw["asset_id"],
                asset_type=AssetType(raw["asset_type"]),
                name=raw["name"],
                version=raw["version"],
                status=LifecycleStage(raw["status"]),
                metadata=raw.get("metadata", {}),
                registered_at=raw.get("registered_at", ""),
                updated_at=raw.get("updated_at")
            )
            self._assets[asset.asset_id] = asset

    def _asset_id(
        self,
        asset_type: AssetType,
        name: str,
        version: str
    ) -> str:
        return f"{asset_type.value}:{name}:{version}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
