from app.config import Settings
from app.config import load_settings
from app.services.rag_service import RAGService


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

    if settings.generation_provider != "extractive":
        raise ServiceConfigurationError(
            "Only GENERATION_PROVIDER=extractive is wired for local runtime. "
            "Inject BedrockAnswerer explicitly when deploying with a "
            "bedrock-runtime client."
        )

    return RAGService()
