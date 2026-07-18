from app.config import Settings
from app.config import load_settings
from app.services.rag_service import RAGService
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.guardrails.base import Guardrail
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.guardrails.llm_judge_hallucination_detector import LLMJudgeHallucinationDetector
from rag.guardrails.manager import GuardrailManager
from rag.guardrails.nli_hallucination_detector import NLIHallucinationDetector
from rag.guardrails.pii_guard import PIIGuard
from rag.guardrails.presidio_pii_guard import PresidioPIIGuard
from rag.retrieval.reranker import CrossEncoderReranker

WIRED_GENERATION_PROVIDERS = ("extractive", "openai_compatible")


class ServiceConfigurationError(ValueError):
    pass


def build_rag_service(
    settings: Settings | None = None
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

    if settings.generation_provider not in WIRED_GENERATION_PROVIDERS:
        raise ServiceConfigurationError(
            "Only GENERATION_PROVIDER=extractive or openai_compatible are wired "
            "for local runtime. Inject BedrockAnswerer explicitly when deploying "
            "with a bedrock-runtime client."
        )

    answerer = None

    if settings.generation_provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_api_key:
            raise ServiceConfigurationError(
                "GENERATION_PROVIDER=openai_compatible requires LLM_BASE_URL "
                "and LLM_API_KEY to be set."
            )

        answerer = OpenAICompatibleAnswerer(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model_name=settings.llm_model_name,
            timeout=settings.llm_timeout_seconds,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature
        )

    reranker = None

    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)

    return RAGService(
        answerer=answerer,
        reranker=reranker,
        candidate_multiplier=settings.reranker_candidate_multiplier,
        guardrail_manager=_build_guardrail_manager(settings)
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
