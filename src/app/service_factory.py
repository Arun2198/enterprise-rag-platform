from app.config import Settings
from app.config import load_settings
from app.services.rag_service import RAGService
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer

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

    if settings.generation_provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_api_key:
            raise ServiceConfigurationError(
                "GENERATION_PROVIDER=openai_compatible requires LLM_BASE_URL "
                "and LLM_API_KEY to be set."
            )

        return RAGService(
            answerer=OpenAICompatibleAnswerer(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model_name=settings.llm_model_name,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature
            )
        )

    return RAGService()
