from app.config import Settings
from app.config import load_settings
from app.services.rag_service import RAGService
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.vector_store.in_memory_store import InMemoryVectorStore
from rag.vector_store.opensearch_client_factory import build_opensearch_client
from rag.vector_store.opensearch_store import OpenSearchVectorStore


class ServiceConfigurationError(ValueError):
    pass


def build_rag_service(
    settings: Settings | None = None
) -> RAGService:
    settings = settings or load_settings()

    if settings.embedding_provider != "hashing":
        raise ServiceConfigurationError(
            "Only EMBEDDING_PROVIDER=hashing is wired for local runtime."
        )

    embedder = HashingEmbedder()
    vector_store = _build_vector_store(
        settings=settings,
        embedding_dimension=embedder.dimensions
    )

    if settings.generation_provider != "extractive":
        raise ServiceConfigurationError(
            "Only GENERATION_PROVIDER=extractive is wired for local runtime. "
            "Inject BedrockAnswerer explicitly when deploying with a "
            "bedrock-runtime client."
        )

    return RAGService(
        embedder=embedder,
        vector_store=vector_store
    )


def _build_vector_store(
    settings: Settings,
    embedding_dimension: int
):
    if settings.vector_store_provider == "memory":
        return InMemoryVectorStore()

    if settings.vector_store_provider == "opensearch":
        client = build_opensearch_client(settings)
        return OpenSearchVectorStore(
            client=client,
            index_name=settings.opensearch_index,
            embedding_dimension=embedding_dimension
        )

    raise ServiceConfigurationError(
        f"Unsupported VECTOR_STORE_PROVIDER={settings.vector_store_provider}"
    )
