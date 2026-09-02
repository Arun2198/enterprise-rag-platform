from app.aws_client_factory import build_boto3_client
from app.config import Settings
from app.config import load_settings
from app.conversation_store import ConversationStore
from app.services.rag_service import RERANKER_FLAG_NAME
from app.services.rag_service import RAGService
from ingestion.manifest_store import InMemoryManifestStore
from ingestion.manifest_store import ManifestStore
from ingestion.manifest_store import S3ManifestStore
from ingestion.s3_document_store import S3DocumentStore
from ingestion.sqs_ingestion_worker import SQSIngestionWorker
from mlops.backup import BackupManager
from mlops.backup import S3BackupTarget
from mlops.feature_flags import FeatureFlagManager
from mlops.ingestion_job_store import IngestionJobStore
from mlops.manager import PlatformManager
from mlops.sqs_scheduler_worker import SQSSchedulerWorker
from rag.embeddings.base import Embedder
from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.embeddings.jina_embedder import JinaEmbedder
from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.generation.base import Answerer
from rag.generation.bedrock_answerer import BedrockAnswerer
from rag.generation.document_first_answerer import DocumentFirstAnswerer
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.generation.fallback_answerer import FallbackAnswerer
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.guardrails.base import Guardrail
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.guardrails.indirect_prompt_injection_guard import IndirectPromptInjectionGuard
from rag.guardrails.llm_judge_hallucination_detector import LLMJudgeHallucinationDetector
from rag.guardrails.manager import GuardrailManager
from rag.guardrails.nli_hallucination_detector import NLIHallucinationDetector
from rag.guardrails.pii_guard import PIIGuard
from rag.guardrails.presidio_pii_guard import PresidioPIIGuard
from rag.guardrails.prompt_injection_guard import PromptInjectionGuard
from rag.guardrails.retrieval_relevance_guard import RetrievalRelevanceGuard
from rag.retrieval.cohere_reranker import CohereReranker
from rag.retrieval.jina_reranker import JinaReranker
from rag.retrieval.reranker import CrossEncoderReranker
from rag.vector_store.base import VectorStore
from rag.vector_store.in_memory_store import InMemoryVectorStore
from rag.vector_store.opensearch_client_factory import build_opensearch_client
from rag.vector_store.opensearch_store import OpenSearchVectorStore

WIRED_GENERATION_PROVIDERS = ("extractive", "openai_compatible", "bedrock")
WIRED_EMBEDDING_PROVIDERS = ("hashing", "sentence_transformer", "jina", "cohere")
WIRED_RERANKER_PROVIDERS = ("local", "jina", "cohere")
WIRED_VECTOR_STORE_PROVIDERS = ("memory", "opensearch")


class ServiceConfigurationError(ValueError):
    pass


def build_rag_service(
    settings: Settings | None = None,
    platform_manager: PlatformManager | None = None
) -> RAGService:
    settings = settings or load_settings()

    embedder = _build_embedder(settings)
    vector_store = _build_vector_store(settings, embedder)
    answerer = _build_answerer(settings.generation_provider, settings, "GENERATION_PROVIDER")

    if settings.generation_fallback_provider:
        fallback_answerer = _build_answerer(
            settings.generation_fallback_provider, settings, "GENERATION_FALLBACK_PROVIDER"
        )
        answerer = FallbackAnswerer(primary=answerer, fallback=fallback_answerer)

    if settings.document_first_answering_enabled and settings.generation_provider != "extractive":
        # Only meaningful when an LLM-backed provider is actually
        # configured - GENERATION_PROVIDER=extractive is already
        # document-only, nothing to route away from. Wraps whatever was
        # built above (including a FallbackAnswerer, if configured) as
        # the "fall back to the LLM" branch, so document-first routing
        # composes with the existing LLM fallback rather than competing
        # with it.
        answerer = DocumentFirstAnswerer(
            document_answerer=ExtractiveAnswerer(),
            llm_answerer=answerer,
            embedder=embedder,
            threshold=settings.retrieval_relevance_threshold
        )

    reranker = _build_reranker(settings) if settings.reranker_enabled else None

    return RAGService(
        embedder=embedder,
        vector_store=vector_store,
        answerer=answerer,
        reranker=reranker,
        candidate_multiplier=settings.reranker_candidate_multiplier,
        feature_flags=_build_feature_flags(settings, platform_manager),
        guardrail_manager=_build_guardrail_manager(settings, embedder),
        ingest_allowed_dir=settings.ingest_allowed_dir,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        rrf_k=settings.rrf_k,
        abstention_enabled=settings.abstention_enabled,
        manifest_store=_build_manifest_store(settings)
    )


def _build_manifest_store(
    settings: Settings
) -> ManifestStore | None:
    """
    None (old, full-re-embed-every-time behavior) when
    INCREMENTAL_INGESTION_ENABLED=false - an explicit opt-out, same
    pattern as every other *_ENABLED flag in this file. Otherwise
    in-process InMemoryManifestStore by default (works with zero config,
    same durability tradeoff as InMemoryRateLimiter - correct for a
    single ECS task, lost on restart/redeploy, not shared across
    replicas), upgraded to durable S3ManifestStore once S3_BUCKET is
    configured, the same opt-in-upgrade pattern as conversations/backups.
    """
    if not settings.incremental_ingestion_enabled:
        return None

    if not settings.s3_bucket:
        return InMemoryManifestStore()

    client = build_boto3_client("s3", region_name=settings.aws_region)
    return S3ManifestStore(
        client=client,
        bucket_name=settings.s3_bucket,
        prefix=settings.ingestion_manifests_s3_prefix
    )


def _build_embedder(
    settings: Settings
) -> Embedder:
    if settings.embedding_provider not in WIRED_EMBEDDING_PROVIDERS:
        raise ServiceConfigurationError(
            f"EMBEDDING_PROVIDER must be one of {WIRED_EMBEDDING_PROVIDERS}, "
            f"got {settings.embedding_provider!r}."
        )

    if settings.embedding_provider == "hashing":
        return HashingEmbedder()

    if settings.embedding_provider == "jina":
        if not settings.jina_api_key:
            raise ServiceConfigurationError(
                "EMBEDDING_PROVIDER=jina requires JINA_API_KEY to be set."
            )

        return JinaEmbedder(
            api_key=settings.jina_api_key,
            model_name=settings.jina_embedding_model,
            dimensions=settings.jina_embedding_dimensions,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries
        )

    if settings.embedding_provider == "cohere":
        if not settings.cohere_api_key:
            raise ServiceConfigurationError(
                "EMBEDDING_PROVIDER=cohere requires COHERE_API_KEY to be set."
            )

        return CohereEmbedder(
            api_key=settings.cohere_api_key,
            model_name=settings.cohere_embedding_model,
            dimensions=settings.cohere_embedding_dimensions,
            timeout=settings.embedding_timeout_seconds,
            max_retries=settings.embedding_max_retries
        )

    return SentenceTransformerEmbedder(model_name=settings.embedding_model_name)


def _build_reranker(
    settings: Settings
):
    if settings.reranker_provider not in WIRED_RERANKER_PROVIDERS:
        raise ServiceConfigurationError(
            f"RERANKER_PROVIDER must be one of {WIRED_RERANKER_PROVIDERS}, "
            f"got {settings.reranker_provider!r}."
        )

    if settings.reranker_provider == "local":
        return CrossEncoderReranker(model_name=settings.reranker_model_name)

    if settings.reranker_provider == "jina":
        if not settings.jina_api_key:
            raise ServiceConfigurationError(
                "RERANKER_PROVIDER=jina requires JINA_API_KEY to be set."
            )

        return JinaReranker(
            api_key=settings.jina_api_key,
            model_name=settings.jina_rerank_model,
            timeout=settings.reranker_timeout_seconds,
            max_retries=settings.reranker_max_retries
        )

    if not settings.cohere_api_key:
        raise ServiceConfigurationError(
            "RERANKER_PROVIDER=cohere requires COHERE_API_KEY to be set."
        )

    return CohereReranker(
        api_key=settings.cohere_api_key,
        model_name=settings.cohere_rerank_model,
        timeout=settings.reranker_timeout_seconds,
        max_retries=settings.reranker_max_retries
    )


def _build_vector_store(
    settings: Settings,
    embedder: Embedder
) -> VectorStore:
    if settings.vector_store_provider not in WIRED_VECTOR_STORE_PROVIDERS:
        raise ServiceConfigurationError(
            f"VECTOR_STORE_PROVIDER must be one of {WIRED_VECTOR_STORE_PROVIDERS}, "
            f"got {settings.vector_store_provider!r}."
        )

    if settings.vector_store_provider == "memory":
        return InMemoryVectorStore()

    if not settings.opensearch_host:
        raise ServiceConfigurationError(
            "VECTOR_STORE_PROVIDER=opensearch requires OPENSEARCH_HOST to be set."
        )

    client = build_opensearch_client(
        host=settings.opensearch_host,
        region=settings.aws_region,
        port=settings.opensearch_port,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        connect_timeout=settings.opensearch_connect_timeout,
        max_retries=settings.opensearch_max_retries
    )
    store = OpenSearchVectorStore(
        client=client,
        index_name=settings.opensearch_index,
        embedding_dimensions=embedder.dimensions
    )
    store.ensure_index(embedder.dimensions)
    return store


def _build_answerer(
    provider: str,
    settings: Settings,
    env_var_name: str
) -> Answerer:
    if provider not in WIRED_GENERATION_PROVIDERS:
        raise ServiceConfigurationError(
            f"{env_var_name} must be one of {WIRED_GENERATION_PROVIDERS}, got {provider!r}."
        )

    if provider == "extractive":
        return ExtractiveAnswerer()

    if provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_api_key:
            raise ServiceConfigurationError(
                f"{env_var_name}=openai_compatible requires LLM_BASE_URL "
                "and LLM_API_KEY to be set."
            )

        return OpenAICompatibleAnswerer(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature
        )

    return BedrockAnswerer(
        client=build_boto3_client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            read_timeout=settings.llm_timeout_seconds
        ),
        model_id=settings.bedrock_model_id,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature
    )


def build_platform_manager(
    settings: Settings | None = None
) -> PlatformManager | None:
    """
    Builds the shared mlops backbone for the live app - just feature flags
    and a scheduler with a default backup job for now (registry/artifacts/
    lifecycle/governance stay available on the instance but nothing in the
    app writes to them yet, same as before this wiring). Returns None when
    MLOPS_ENABLED=false so callers can skip mlops entirely rather than
    holding an inert instance.
    """
    settings = settings or load_settings()

    if not settings.mlops_enabled:
        return None

    manager = PlatformManager(
        backup=BackupManager(
            output_dir=settings.scheduler_backup_dir,
            target=_build_backup_target(settings)
        )
    )

    if settings.feature_flags_enabled:
        _define_reranker_flag(manager.feature_flags, settings)

    if settings.scheduler_enabled:
        manager.scheduler.register(
            job_id="backup",
            name="Platform state backup",
            interval_seconds=settings.scheduler_interval_seconds,
            callable_=manager.create_backup
        )

    return manager


def _build_backup_target(
    settings: Settings
) -> S3BackupTarget | None:
    """
    Returns None (local-file-only backup) when S3_BUCKET isn't set -
    same opt-in pattern as async ingestion. When it is set, MLOps
    platform state (registry/artifacts/configuration/feature_flags)
    survives an ECS task restart instead of vanishing with the
    container's local disk, reusing the same bucket async ingestion
    already uses (a distinct prefix, no new bucket to provision/pay
    for).
    """
    if not settings.s3_bucket:
        return None

    client = build_boto3_client("s3", region_name=settings.aws_region)
    return S3BackupTarget(
        client=client,
        bucket_name=settings.s3_bucket,
        prefix=settings.mlops_backup_s3_prefix
    )


def _build_feature_flags(
    settings: Settings,
    platform_manager: PlatformManager | None
) -> FeatureFlagManager | None:
    if not settings.feature_flags_enabled:
        return None

    manager = platform_manager.feature_flags if platform_manager is not None else FeatureFlagManager()
    _define_reranker_flag(manager, settings)
    return manager


def _define_reranker_flag(
    manager: FeatureFlagManager,
    settings: Settings
) -> None:
    if any(flag.name == RERANKER_FLAG_NAME for flag in manager.list()):
        return

    manager.define(
        RERANKER_FLAG_NAME,
        enabled=True,
        rollout_percentage=settings.reranker_rollout_percentage,
        description="Percentage of /ask requests that get cross-encoder reranking"
    )


def _build_guardrail_manager(
    settings: Settings,
    embedder: Embedder
) -> GuardrailManager:
    if not settings.guardrails_enabled:
        return GuardrailManager(guardrails=[])

    guardrails: list[Guardrail] = []

    if settings.prompt_injection_guard_enabled:
        guardrails.append(PromptInjectionGuard())

    if settings.indirect_prompt_injection_guard_enabled:
        guardrails.append(IndirectPromptInjectionGuard())

    if settings.pii_guard_enabled:
        guardrails.append(PIIGuard())

    if settings.presidio_pii_guard_enabled:
        guardrails.append(
            PresidioPIIGuard(
                entities=settings.presidio_entities,
                score_threshold=settings.presidio_score_threshold
            )
        )

    if settings.hallucination_guard_enabled:
        guardrails.append(
            HallucinationDetector(
                threshold=settings.groundedness_threshold,
                embedder=embedder
            )
        )

    if settings.retrieval_relevance_guard_enabled:
        # Off by default - see RetrievalRelevanceGuard's module docstring
        # and default_retrieval_relevance_threshold: the auto-picked
        # threshold is only calibrated against a real dense embedder
        # (verified zero false positives on the golden dataset with
        # BAAI/bge-small-en-v1.5); HashingEmbedder does not separate
        # relevant from irrelevant queries reliably enough for this to be
        # a safe default everywhere.
        guardrails.append(
            RetrievalRelevanceGuard(
                embedder=embedder,
                threshold=settings.retrieval_relevance_threshold
            )
        )

    if settings.nli_hallucination_enabled:
        guardrails.append(
            NLIHallucinationDetector(
                model_name=settings.nli_model_name,
                threshold=settings.nli_threshold
            )
        )

    if settings.llm_judge_enabled:
        base_url = settings.llm_judge_base_url or settings.llm_base_url
        api_key = settings.llm_judge_api_key or settings.llm_api_key

        if not base_url or not api_key:
            raise ServiceConfigurationError(
                "LLM_JUDGE_ENABLED=true requires LLM_JUDGE_BASE_URL (or "
                "LLM_BASE_URL) and LLM_JUDGE_API_KEY (or LLM_API_KEY) to be set."
            )

        guardrails.append(
            LLMJudgeHallucinationDetector(
                api_key=api_key,
                base_url=base_url,
                model_name=settings.llm_judge_model_name or settings.llm_model_name,
                threshold=settings.llm_judge_threshold
            )
        )

    return GuardrailManager(guardrails=guardrails)


def build_s3_document_store(
    settings: Settings
) -> S3DocumentStore | None:
    """
    Returns None (not an error) when S3_BUCKET isn't set - async
    ingestion is opt-in infrastructure, not something every deployment
    needs; the synchronous local-file /ingest path works with zero S3
    configuration at all.
    """
    if not settings.s3_bucket:
        return None

    client = build_boto3_client("s3", region_name=settings.aws_region)
    return S3DocumentStore(
        client=client,
        bucket_name=settings.s3_bucket,
        raw_prefix=settings.s3_raw_prefix,
        processed_prefix=settings.s3_processed_prefix,
        failed_prefix=settings.s3_failed_prefix,
        max_file_size_bytes=settings.s3_max_file_size_mb * 1024 * 1024
    )


def build_ingestion_job_store(
    settings: Settings
) -> IngestionJobStore | None:
    if not settings.s3_bucket:
        return None

    client = build_boto3_client("s3", region_name=settings.aws_region)
    return IngestionJobStore(
        client=client,
        bucket_name=settings.s3_bucket,
        prefix=settings.s3_jobs_prefix
    )


def build_conversation_store(
    settings: Settings
) -> ConversationStore | None:
    """
    Returns None when S3_BUCKET isn't set - same opt-in pattern as async
    ingestion and durable backups. Chat memory only works when there's
    somewhere durable to put it; without S3_BUCKET the /ask route just
    falls back to stateless single-turn behavior (see main.py).
    """
    if not settings.s3_bucket:
        return None

    client = build_boto3_client("s3", region_name=settings.aws_region)
    return ConversationStore(
        client=client,
        bucket_name=settings.s3_bucket,
        prefix=settings.conversations_s3_prefix
    )


def build_sqs_client(
    settings: Settings
):
    """
    None when SQS isn't configured. Shared by the upload endpoint
    (send_message, to enqueue a job) and the ingestion worker
    (receive_message/delete_message, to process one) - one client, one
    connection pool, rather than each building its own.
    """
    if not settings.sqs_queue_url:
        return None

    return build_boto3_client("sqs", region_name=settings.aws_region)


def build_sqs_ingestion_worker(
    settings: Settings,
    rag_service: RAGService,
    s3_store: S3DocumentStore,
    job_store: IngestionJobStore,
    sqs_client
) -> SQSIngestionWorker | None:
    """
    Returns None when async ingestion isn't fully configured
    (ASYNC_INGESTION_ENABLED + SQS_QUEUE_URL + S3_BUCKET all required) -
    the API and its synchronous /ingest path work identically either way.
    """
    if not settings.async_ingestion_enabled or sqs_client is None:
        return None

    # build_sqs_client() only ever returns non-None when sqs_queue_url is
    # set (see its own guard) - sqs_client not being None here proves it.
    assert settings.sqs_queue_url is not None

    return SQSIngestionWorker(
        sqs_client=sqs_client,
        queue_url=settings.sqs_queue_url,
        ingestion_pipeline=rag_service.ingestion_pipeline,
        s3_store=s3_store,
        job_store=job_store,
        rag_service=rag_service
    )


def build_scheduler_sqs_client(
    settings: Settings
):
    """
    None when SCHEDULER_QUEUE_URL isn't set - a separate client/queue
    from build_sqs_client's ingestion queue, since these are two
    unrelated message flows (document ingestion vs. EventBridge-driven
    scheduled jobs) that shouldn't share a queue's visibility timeout/
    redrive policy.
    """
    if not settings.scheduler_queue_url:
        return None

    return build_boto3_client("sqs", region_name=settings.aws_region)


def build_scheduler_sqs_worker(
    settings: Settings,
    platform_manager: PlatformManager | None,
    sqs_client
) -> SQSSchedulerWorker | None:
    """
    Returns None unless MLOps + the scheduler + SCHEDULER_QUEUE_URL are
    all configured - main.py falls back to the plain in-process interval
    loop (Scheduler.run_due_jobs() on a sleep loop) when this is None,
    exactly the pre-existing behavior. Once set, EventBridge Scheduler
    (see terraform/) becomes the source of truth for *when* each job
    runs, and this worker becomes the thing that actually executes it -
    SQS's single-delivery-per-consumer guarantee is what prevents every
    ECS task in the service from firing the same job.
    """
    if (
        platform_manager is None
        or not settings.scheduler_enabled
        or sqs_client is None
    ):
        return None

    assert settings.scheduler_queue_url is not None

    return SQSSchedulerWorker(
        sqs_client=sqs_client,
        queue_url=settings.scheduler_queue_url,
        scheduler=platform_manager.scheduler
    )
