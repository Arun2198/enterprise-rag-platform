import logging
from typing import Any

from mlops import telemetry
from mlops.artifacts import ArtifactRegistry
from mlops.backup import BackupManager
from mlops.configuration import ConfigurationManager
from mlops.feature_flags import FeatureFlagManager
from mlops.governance import GovernanceLog
from mlops.lifecycle import LifecycleManager
from mlops.recovery import RecoveryManager
from mlops.registry import ModelRegistry
from mlops.schemas import AssetType
from mlops.schemas import BackupSnapshot
from mlops.schemas import LifecycleStage
from mlops.schemas import ModelAsset
from mlops.scheduler import Scheduler
from mlops.secrets import LocalEnvSecretsProvider
from mlops.secrets import SecretsProvider

logger = logging.getLogger(__name__)


class ProviderNotFoundError(KeyError):
    pass


class PlatformManager:
    """
    Facade coordinating the MLOps platform's components. Each component
    is independently usable on its own (import ModelRegistry directly if
    that's all a caller needs) - PlatformManager exists for
    cross-component workflows like "promote" (a lifecycle transition
    plus a mirrored governance record, in one call) and to give provider
    adapters (a real MLflow registry backend, a real cloud secrets
    provider, a real CI deployment pipeline, a drift detector, ...) one
    place to register via register_provider(), keyed by name.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        artifacts: ArtifactRegistry | None = None,
        configuration: ConfigurationManager | None = None,
        feature_flags: FeatureFlagManager | None = None,
        secrets: SecretsProvider | None = None,
        scheduler: Scheduler | None = None,
        governance: GovernanceLog | None = None,
        backup: BackupManager | None = None,
        recovery: RecoveryManager | None = None
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.artifacts = artifacts or ArtifactRegistry()
        self.configuration = configuration or ConfigurationManager()
        self.feature_flags = feature_flags or FeatureFlagManager()
        self.secrets = secrets or LocalEnvSecretsProvider()
        self.scheduler = scheduler or Scheduler()
        self.governance = governance or GovernanceLog()
        self.backup = backup or BackupManager()
        self.recovery = recovery or RecoveryManager()
        self.lifecycle = LifecycleManager(self.registry)
        self._providers: dict[str, Any] = {}

    def register_provider(
        self,
        name: str,
        provider: Any
    ) -> None:
        """
        Register a pluggable backend - a real MLflow ModelRegistryBackend,
        a cloud SecretsProvider, a CI DeploymentPipeline, a
        DriftDetector, a RetrainingTrigger, a BackupTarget, etc.
        PlatformManager doesn't care what shape it is; this is just a
        named slot other code can look up via get_provider().
        """
        self._providers[name] = provider
        logger.info("provider_registered", extra={"provider": name})

    def get_provider(
        self,
        name: str
    ) -> Any:
        if name not in self._providers:
            raise ProviderNotFoundError(name)

        return self._providers[name]

    def register_asset(
        self,
        asset_type: AssetType,
        name: str,
        version: str,
        metadata: dict[str, Any] | None = None
    ) -> ModelAsset:
        asset = self.registry.register(
            asset_type=asset_type,
            name=name,
            version=version,
            metadata=metadata
        )
        self.governance.record(
            actor=None,
            action="register_asset",
            resource=asset.asset_id,
            details={"asset_type": asset_type.value, "version": version}
        )
        telemetry.record_operation("register_asset", success=True)
        telemetry.record_audit_event("register_asset")
        return asset

    def promote(
        self,
        asset_id: str,
        to_stage: LifecycleStage,
        actor: str | None = None,
        reason: str | None = None,
        approved_by: str | None = None
    ) -> ModelAsset:
        try:
            asset = self.lifecycle.promote(
                asset_id,
                to_stage,
                actor=actor,
                reason=reason,
                approved_by=approved_by
            )
        except Exception:
            telemetry.record_operation("promote", success=False)
            raise

        transition = self.lifecycle.history(asset_id)[-1]
        self.governance.record_transition(transition)
        telemetry.record_audit_event("lifecycle_transition")

        if approved_by is not None:
            approval = self.lifecycle.approvals(asset_id)[-1]
            self.governance.record_approval(approval)
            telemetry.record_audit_event("approval")

        telemetry.record_operation("promote", success=True)
        return asset

    def link_asset_lineage(
        self,
        asset_id: str,
        artifact_id: str,
        artifact_version: int
    ) -> None:
        self.governance.link_lineage(asset_id, artifact_id, artifact_version)
        telemetry.record_audit_event("lineage_link")

    def create_backup(self) -> BackupSnapshot:
        snapshot = self.backup.create_snapshot({
            "registry": self.registry,
            "artifacts": self.artifacts,
            "configuration": self.configuration,
            "feature_flags": self.feature_flags,
        })
        self.governance.record(
            actor=None,
            action="backup_created",
            resource=snapshot.snapshot_id,
            details={"components": snapshot.components}
        )
        telemetry.record_operation("backup", success=True)
        telemetry.record_audit_event("backup_created")
        return snapshot

    def restore_backup(
        self,
        path: str
    ) -> list[str]:
        restored = self.recovery.restore_snapshot(path, {
            "registry": self.registry,
            "artifacts": self.artifacts,
            "configuration": self.configuration,
            "feature_flags": self.feature_flags,
        })
        self.governance.record(
            actor=None,
            action="backup_restored",
            resource=path,
            details={"components": restored}
        )
        telemetry.record_operation("restore", success=True)
        telemetry.record_audit_event("backup_restored")
        return restored
