import boto3

from app.config import Settings
from app.config import load_settings
from app.services.rag_service import RERANKER_FLAG_NAME
from app.services.rag_service import RAGService
from mlops.backup import BackupManager
from mlops.feature_flags import FeatureFlagManager
from mlops.manager import PlatformManager
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.generation.base import Answerer
from rag.generation.bedrock_answerer import BedrockAnswerer
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.generation.fallback_answerer import FallbackAnswerer
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.guardrails.base import Guardrail
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.guardrails.llm_judge_hallucination_detector import LLMJudgeHallucinationDetector
from rag.guardrails.manager import GuardrailManager
from rag.guardrails.nli_hallucination_detector import NLIHallucinationDetector
from rag.guardrails.pii_guard import PIIGuard
from rag.guardrails.presidio_pii_guard import PresidioPIIGuard
from rag.retrieval.reranker import CrossEncoderReranker

WIRED_GENERATION_PROVIDERS = ("extractive", "openai_compatible", "bedrock")


class ServiceConfigurationError(ValueError):
    pass


def build_rag_service(
    settings: Settings | None = None,
    platform_manager: PlatformManager | None = None
) -> RAGService:
    settings = settings or load_settings()

    if settings.vector_store_provider != "memory":
        raise ServiceConfigurationError(
            "Only VECTOR_STORE_PROVIDER=memory is wired for local runtime. "
            "Inject OpenSearchVectorStore explicitly when deploying with an "
            "authenticated OpenSearch client."
        )

    if settings.embedding_provider != "hashing":
        raise ServiceConfigurationError(
            "Only EMBEDDING_PROVIDER=hashing is wired for local runtime."
        )

    answerer = _build_answerer(settings.generation_provider, settings, "GENERATION_PROVIDER")

    if settings.generation_fallback_provider:
        fallback_answerer = _build_answerer(
            settings.generation_fallback_provider, settings, "GENERATION_FALLBACK_PROVIDER"
        )
        answerer = FallbackAnswerer(primary=answerer, fallback=fallback_answerer)

    reranker = None

    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)

    return RAGService(
        answerer=answerer,
        reranker=reranker,
        candidate_multiplier=settings.reranker_candidate_multiplier,
        feature_flags=_build_feature_flags(settings, platform_manager),
        guardrail_manager=_build_guardrail_manager(settings)
    )


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
        client=boto3.client("bedrock-runtime", region_name=settings.aws_region),
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
        backup=BackupManager(output_dir=settings.scheduler_backup_dir)
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
    settings: Settings
) -> GuardrailManager:
    if not settings.guardrails_enabled:
        return GuardrailManager(guardrails=[])

    guardrails: list[Guardrail] = []

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
                embedder=HashingEmbedder()
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
