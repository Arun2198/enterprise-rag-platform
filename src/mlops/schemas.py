from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


class AssetType(str, Enum):
    EMBEDDING_MODEL = "embedding_model"
    RERANKER = "reranker"
    LLM_PROVIDER = "llm_provider"
    PROMPT_TEMPLATE = "prompt_template"
    GUARDRAIL_MODEL = "guardrail_model"
    EVALUATION_MODEL = "evaluation_model"


class LifecycleStage(str, Enum):
    """
    Shared by ModelRegistry (an asset's current status) and
    LifecycleManager (the promotion state machine). Registry status is
    just "which stage is this asset currently in" using the same values
    the lifecycle state machine transitions between.
    """
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    RETIRED = "retired"


class Role(str, Enum):
    ADMINISTRATOR = "administrator"
    ML_ENGINEER = "ml_engineer"
    DATA_SCIENTIST = "data_scientist"
    REVIEWER = "reviewer"
    READ_ONLY = "read_only"


class Permission(str, Enum):
    VIEW = "view"
    REGISTER_ASSET = "register_asset"
    PROMOTE_ASSET = "promote_asset"
    APPROVE_PROMOTION = "approve_promotion"
    RETIRE_ASSET = "retire_asset"
    EDIT_CONFIGURATION = "edit_configuration"
    TOGGLE_FEATURE_FLAG = "toggle_feature_flag"
    TRIGGER_DEPLOYMENT = "trigger_deployment"
    TRIGGER_RETRAINING = "trigger_retraining"
    MANAGE_SECRETS = "manage_secrets"
    TRIGGER_BACKUP = "trigger_backup"
    TRIGGER_RESTORE = "trigger_restore"


@dataclass(frozen=True)
class ModelAsset:
    asset_id: str
    asset_type: AssetType
    name: str
    version: str
    status: LifecycleStage
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: str = ""
    updated_at: str | None = None


@dataclass(frozen=True)
class ArtifactVersion:
    artifact_id: str
    artifact_type: str
    version: int
    content: dict[str, Any]
    created_at: str
    created_by: str | None = None


@dataclass(frozen=True)
class LifecycleTransition:
    asset_id: str
    from_stage: LifecycleStage | None
    to_stage: LifecycleStage
    timestamp: str
    actor: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class ApprovalRecord:
    asset_id: str
    to_stage: LifecycleStage
    approved_by: str
    timestamp: str
    comment: str | None = None


@dataclass(frozen=True)
class ConfigProfile:
    name: str
    values: dict[str, Any]
    version: int
    created_at: str
    description: str | None = None


@dataclass(frozen=True)
class FeatureFlag:
    name: str
    enabled: bool
    rollout_percentage: float = 100.0
    shadow: bool = False
    description: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class ScheduledJob:
    job_id: str
    name: str
    interval_seconds: float
    next_run_at: float
    enabled: bool = True


@dataclass(frozen=True)
class JobRun:
    job_id: str
    started_at: str
    finished_at: str | None
    success: bool | None
    error: str | None = None


@dataclass(frozen=True)
class BackupSnapshot:
    snapshot_id: str
    created_at: str
    components: list[str]
    path: str


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    timestamp: str
    actor: str | None
    action: str
    resource: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriftReport:
    drift_type: str
    detected: bool
    score: float
    threshold: float
    timestamp: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrainingRequest:
    asset_id: str
    trigger: str
    requested_at: str
    requested_by: str | None = None
    reason: str | None = None
