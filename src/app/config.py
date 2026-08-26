import os
from dataclasses import dataclass


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in ("false", "0", "no", "")


@dataclass(frozen=True)
class Settings:
    vector_store_provider: str = "memory"
    embedding_provider: str = "sentence_transformer"
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    jina_api_key: str | None = None
    jina_embedding_model: str = "jina-embeddings-v3"
    jina_embedding_dimensions: int = 1024
    cohere_api_key: str | None = None
    cohere_embedding_model: str = "embed-english-v3.0"
    cohere_embedding_dimensions: int = 1024
    embedding_timeout_seconds: float = 30.0
    embedding_max_retries: int = 3
    ingest_allowed_dir: str = "sample_documents"
    generation_provider: str = "extractive"
    generation_fallback_provider: str | None = None
    opensearch_host: str | None = None
    opensearch_index: str = "enterprise-rag-chunks"
    opensearch_port: int = 443
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = True
    opensearch_connect_timeout: float = 10.0
    opensearch_max_retries: int = 3
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    s3_bucket: str | None = None
    s3_raw_prefix: str = "raw/"
    s3_processed_prefix: str = "processed/"
    s3_failed_prefix: str = "failed/"
    s3_max_file_size_mb: int = 25
    s3_jobs_prefix: str = "jobs/"
    sqs_queue_url: str | None = None
    sqs_poll_interval_seconds: float = 20.0
    async_ingestion_enabled: bool = False
    auth_enabled: bool = False
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_role_claim: str = "role"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    reranker_enabled: bool = True
    reranker_provider: str = "local"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidate_multiplier: int = 4
    jina_rerank_model: str = "jina-reranker-v2-base-multilingual"
    cohere_rerank_model: str = "rerank-english-v3.0"
    reranker_timeout_seconds: float = 30.0
    reranker_max_retries: int = 3
    guardrails_enabled: bool = True
    prompt_injection_guard_enabled: bool = True
    indirect_prompt_injection_guard_enabled: bool = True
    pii_guard_enabled: bool = True
    hallucination_guard_enabled: bool = True
    groundedness_threshold: float = 0.60
    retrieval_relevance_guard_enabled: bool = False
    retrieval_relevance_threshold: float | None = None
    cors_allowed_origins: tuple[str, ...] = ()
    cloudwatch_metrics_enabled: bool = False
    cloudwatch_metrics_namespace: str = "EnterpriseRAGPlatform"
    cloudwatch_metrics_export_interval_seconds: float = 60.0
    abstention_enabled: bool = True
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
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "sentence_transformer"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"),
        jina_api_key=os.getenv("JINA_API_KEY"),
        jina_embedding_model=os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"),
        jina_embedding_dimensions=int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "1024")),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        cohere_embedding_model=os.getenv("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"),
        cohere_embedding_dimensions=int(os.getenv("COHERE_EMBEDDING_DIMENSIONS", "1024")),
        embedding_timeout_seconds=float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "30")),
        embedding_max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "3")),
        ingest_allowed_dir=os.getenv("INGEST_ALLOWED_DIR", "sample_documents"),
        generation_provider=os.getenv("GENERATION_PROVIDER", "extractive"),
        generation_fallback_provider=os.getenv("GENERATION_FALLBACK_PROVIDER"),
        opensearch_host=os.getenv("OPENSEARCH_HOST"),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "enterprise-rag-chunks"),
        opensearch_port=int(os.getenv("OPENSEARCH_PORT", "443")),
        opensearch_use_ssl=_parse_bool(os.getenv("OPENSEARCH_USE_SSL", "true")),
        opensearch_verify_certs=_parse_bool(os.getenv("OPENSEARCH_VERIFY_CERTS", "true")),
        opensearch_connect_timeout=float(os.getenv("OPENSEARCH_CONNECT_TIMEOUT", "10")),
        opensearch_max_retries=int(os.getenv("OPENSEARCH_MAX_RETRIES", "3")),
        dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
        rrf_k=int(os.getenv("RRF_K", "60")),
        s3_bucket=os.getenv("S3_BUCKET"),
        s3_raw_prefix=os.getenv("S3_RAW_PREFIX", "raw/"),
        s3_processed_prefix=os.getenv("S3_PROCESSED_PREFIX", "processed/"),
        s3_failed_prefix=os.getenv("S3_FAILED_PREFIX", "failed/"),
        s3_max_file_size_mb=int(os.getenv("S3_MAX_FILE_SIZE_MB", "25")),
        s3_jobs_prefix=os.getenv("S3_JOBS_PREFIX", "jobs/"),
        sqs_queue_url=os.getenv("SQS_QUEUE_URL"),
        sqs_poll_interval_seconds=float(os.getenv("SQS_POLL_INTERVAL_SECONDS", "20")),
        async_ingestion_enabled=_parse_bool(os.getenv("ASYNC_INGESTION_ENABLED", "false")),
        auth_enabled=_parse_bool(os.getenv("AUTH_ENABLED", "false")),
        oidc_issuer=os.getenv("OIDC_ISSUER"),
        oidc_audience=os.getenv("OIDC_AUDIENCE"),
        oidc_jwks_url=os.getenv("OIDC_JWKS_URL"),
        oidc_role_claim=os.getenv("OIDC_ROLE_CLAIM", "role"),
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
        reranker_provider=os.getenv("RERANKER_PROVIDER", "local"),
        jina_rerank_model=os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual"),
        cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        reranker_timeout_seconds=float(os.getenv("RERANKER_TIMEOUT_SECONDS", "30")),
        reranker_max_retries=int(os.getenv("RERANKER_MAX_RETRIES", "3")),
        guardrails_enabled=_parse_bool(os.getenv("GUARDRAILS_ENABLED", "true")),
        prompt_injection_guard_enabled=_parse_bool(
            os.getenv("PROMPT_INJECTION_GUARD_ENABLED", "true")
        ),
        indirect_prompt_injection_guard_enabled=_parse_bool(
            os.getenv("INDIRECT_PROMPT_INJECTION_GUARD_ENABLED", "true")
        ),
        pii_guard_enabled=_parse_bool(os.getenv("PII_GUARD_ENABLED", "true")),
        hallucination_guard_enabled=_parse_bool(
            os.getenv("HALLUCINATION_GUARD_ENABLED", "true")
        ),
        groundedness_threshold=float(os.getenv("GROUNDEDNESS_THRESHOLD", "0.60")),
        retrieval_relevance_guard_enabled=_parse_bool(
            os.getenv("RETRIEVAL_RELEVANCE_GUARD_ENABLED", "false")
        ),
        retrieval_relevance_threshold=(
            float(_retrieval_relevance_threshold_raw)
            if (_retrieval_relevance_threshold_raw := os.getenv("RETRIEVAL_RELEVANCE_THRESHOLD"))
            else None
        ),
        cors_allowed_origins=(
            tuple(origin.strip() for origin in _cors_origins_raw.split(","))
            if (_cors_origins_raw := os.getenv("CORS_ALLOWED_ORIGINS")) else ()
        ),
        cloudwatch_metrics_enabled=_parse_bool(os.getenv("CLOUDWATCH_METRICS_ENABLED", "false")),
        cloudwatch_metrics_namespace=os.getenv(
            "CLOUDWATCH_METRICS_NAMESPACE", "EnterpriseRAGPlatform"
        ),
        cloudwatch_metrics_export_interval_seconds=float(
            os.getenv("CLOUDWATCH_METRICS_EXPORT_INTERVAL_SECONDS", "60")
        ),
        abstention_enabled=_parse_bool(os.getenv("ABSTENTION_ENABLED", "true")),
        presidio_pii_guard_enabled=_parse_bool(
            os.getenv("PRESIDIO_PII_GUARD_ENABLED", "false")
        ),
        presidio_score_threshold=float(os.getenv("PRESIDIO_SCORE_THRESHOLD", "0.5")),
        presidio_entities=(
            tuple(_presidio_entities_raw.split(","))
            if (_presidio_entities_raw := os.getenv("PRESIDIO_ENTITIES")) else None
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
