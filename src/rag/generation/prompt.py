from rag.retrieval.hybrid_retrieval import RetrievedChunk


def build_grounded_prompt(
    query: str,
    retrieved_chunks: list[RetrievedChunk]
) -> str:
    """
    Shared grounded-answer prompt used by every LLM-backed Answerer, so all
    providers answer only from retrieved context and cite the same source ids.
    """
    context = "\n\n".join(
        f"Source {index + 1} ({item.chunk.chunk_id}):\n{item.chunk.text}"
        for index, item in enumerate(retrieved_chunks)
    )

    return (
        "Answer the question using only the provided context. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
