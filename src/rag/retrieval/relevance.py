from rag.embeddings.base import Embedder
from rag.retrieval.hybrid_retrieval import RetrievedChunk


def cosine_similarity(
    first: list[float],
    second: list[float]
) -> float:
    numerator = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = sum(a * a for a in first) ** 0.5
    second_norm = sum(b * b for b in second) ** 0.5

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return numerator / (first_norm * second_norm)


def best_retrieval_relevance(
    embedder: Embedder,
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    top_n: int = 3
) -> float:
    """
    Cosine similarity between the query and its best-matching retrieved
    chunk, both freshly embedded through the given Embedder - independent
    of whatever scoring scale the retriever/vector store reports (see
    RetrievalRelevanceGuard's module docstring for why). Shared by that
    guard and DocumentFirstAnswerer's routing decision so both use
    exactly the same signal rather than two subtly different ones.
    """
    query_embedding = embedder.embed(query)
    return max(
        cosine_similarity(query_embedding, embedder.embed(item.chunk.text))
        for item in retrieved_chunks[:top_n]
    )
