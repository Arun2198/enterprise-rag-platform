import os
from dataclasses import dataclass


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in ("false", "0", "no", "")


@dataclass(frozen=True)
class Settings:
    vector_store_provider: str = "memory"
    embedding_provider: str = "hashing"
    generation_provider: str = "extractive"
    opensearch_host: str | None = None
    opensearch_index: str = "enterprise-rag-chunks"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidate_multiplier: int = 4
    guardrails_enabled: bool = True
    pii_guard_enabled: bool = True
    hallucination_guard_enabled: bool = True
    groundedness_threshold: float = 0.60
    presidio_pii_guard_enabled: bool = False
    presidio_score_threshold: float = 0.5
    presidio_entities: tuple[str, ...] | None = None
    nli_hallucination_enabled: bool = False
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    nli_threshold: float = 0.50
    llm_judge_enabled: bool = False
    llm_judge_base_url: str | None = None
    llm_judge_api_key: str | None = None
    llm_judge_model_name: str | None = None
    llm_judge_threshold: float = 0.60
    evaluation_enabled: bool = True
    evaluation_default_k: int = 5
    evaluation_report_dir: str = "evaluation/reports"
    mlops_enabled: bool = True
    model_registry_enabled: bool = True
    feature_flags_enabled: bool = True
    secrets_enabled: bool = True
    drift_monitoring_enabled: bool = True
    scheduler_enabled: bool = True
    reranker_rollout_percentage: float = 100.0
    scheduler_interval_seconds: float = 300.0
    scheduler_backup_dir: str = "mlops_backups"


def load_settings() -> Settings:
    return Settings(
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "memory"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing"),
        generation_provider=os.getenv("GENERATION_PROVIDER", "extractive"),
        opensearch_host=os.getenv("OPENSEARCH_HOST"),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "enterprise-rag-chunks"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        reranker_enabled=_parse_bool(os.getenv("RERANKER_ENABLED", "true")),
        reranker_model_name=os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        reranker_candidate_multiplier=int(
            os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "4")
        ),
        guardrails_enabled=_parse_bool(os.getenv("GUARDRAILS_ENABLED", "true")),
        pii_guard_enabled=_parse_bool(os.getenv("PII_GUARD_ENABLED", "true")),
        hallucination_guard_enabled=_parse_bool(
            os.getenv("HALLUCINATION_GUARD_ENABLED", "true")
        ),
        groundedness_threshold=float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.60")),
        presidio_pii_guard_enabled=_parse_bool(
            os.getenv("PRESIDIO_PII_GUARD_ENABLED", "false")
        ),
        presidio_score_threshold=float(os.getenv("PRESIDIO_SCORE_THRESHOLD", "0.5")),
        presidio_entities=(
            tuple(os.getenv("PRESIDIO_ENTITIES").split(","))
            if os.getenv("PRESIDIO_ENTITIES") else None
        ),
        nli_hallucination_enabled=_parse_bool(
            os.getenv("NLI_HALLUCINATION_ENABLED", "false")
        ),
        nli_model_name=os.getenv(
            "NLI_MODEL_NAME",
            "cross-encoder/nli-deberta-v3-base"
        ),
        nli_threshold=float(os.getenv("NLI_THRESHOLD", "0.50")),
        llm_judge_enabled=_parse_bool(os.getenv("LLM_JUDGE_ENABLED", "false")),
        llm_judge_base_url=os.getenv("LLM_JUDGE_BASE_URL"),
        llm_judge_api_key=os.getenv("LLM_JUDGE_API_KEY"),
        llm_judge_model_name=os.getenv("LLM_JUDGE_MODEL_NAME"),
        llm_judge_threshold=float(os.getenv("LLM_JUDGE_THRESHOLD", "0.60")),
        evaluation_enabled=_parse_bool(os.getenv("EVALUATION_ENABLED", "true")),
        evaluation_default_k=int(os.getenv("EVALUATION_DEFAULT_K", "5")),
        evaluation_report_dir=os.getenv("EVALUATION_REPORT_DIR", "evaluation/reports"),
        mlops_enabled=_parse_bool(os.getenv("MLOPS_ENABLED", "true")),
        model_registry_enabled=_parse_bool(os.getenv("MODEL_REGISTRY_ENABLED", "true")),
        feature_flags_enabled=_parse_bool(os.getenv("FEATURE_FLAGS_ENABLED", "true")),
        secrets_enabled=_parse_bool(os.getenv("SECRETS_ENABLED", "true")),
        drift_monitoring_enabled=_parse_bool(os.getenv("DRIFT_MONITORING_ENABLED", "true")),
        scheduler_enabled=_parse_bool(os.getenv("SCHEDULER_ENABLED", "true")),
        reranker_rollout_percentage=float(os.getenv("RERANKER_ROLLOUT_PERCENTAGE", "100")),
        scheduler_interval_seconds=float(os.getenv("SCHEDULER_INTERVAL_SECONDS", "300")),
        scheduler_backup_dir=os.getenv("SCHEDULER_BACKUP_DIR", "mlops_backups"),
    )
