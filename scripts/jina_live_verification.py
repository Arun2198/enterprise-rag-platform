"""
One-off live verification that JinaEmbedder/JinaReranker actually work
against the real Jina API - not mocked HTTP. Requires JINA_API_KEY in
the environment (e.g. via a local .env, gitignored). Prints only
verification results, never the key.

Run: set -a; source .env; set +a; uv run python scripts/jina_live_verification.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag.chunking.chunk import Chunk  # noqa: E402
from rag.embeddings.jina_embedder import JinaEmbedder  # noqa: E402
from rag.retrieval.hybrid_retrieval import RetrievedChunk  # noqa: E402
from rag.retrieval.jina_reranker import JinaReranker  # noqa: E402


def main() -> int:
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        print("JINA_API_KEY not set - nothing to verify.", file=sys.stderr)
        return 1

    print("=== JinaEmbedder ===")
    embedder = JinaEmbedder(api_key=api_key)
    vectors = embedder.embed_batch(["Water boils at 100 degrees Celsius.", "Paris is the capital of France."])
    print(f"provider_name={embedder.provider_name} model={embedder.model_name}")
    print(f"batch size={len(vectors)} dims={[len(v) for v in vectors]}")
    assert len(vectors) == 2
    assert all(len(v) == embedder.dimensions for v in vectors)
    print("PASS: batch embedding call succeeded with expected shape\n")

    print("=== JinaReranker ===")
    reranker = JinaReranker(api_key=api_key)
    relevant = Chunk(
        chunk_id="doc:0", document_id="doc", chunk_index=0,
        text="Water boils at 100 degrees Celsius at sea level.",
        source="doc.md", document_type="markdown"
    )
    irrelevant = Chunk(
        chunk_id="doc:1", document_id="doc", chunk_index=1,
        text="The Eiffel Tower is a famous landmark in Paris.",
        source="doc.md", document_type="markdown"
    )
    candidates = [
        RetrievedChunk(chunk=irrelevant, vector_score=0.5, keyword_score=0.5, score=0.5),
        RetrievedChunk(chunk=relevant, vector_score=0.5, keyword_score=0.5, score=0.5),
    ]
    reranked = reranker.rerank(
        query="What temperature does water boil at?",
        candidates=candidates,
        top_k=2
    )
    for item in reranked:
        print(f"rank={item.rank} score={item.score:.4f} text={item.chunk.text[:50]!r}")
    assert reranked[0].chunk.chunk_id == "doc:0", "relevant chunk should rank first"
    print("PASS: reranker correctly ranked the relevant chunk first\n")

    print("All live verifications passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
