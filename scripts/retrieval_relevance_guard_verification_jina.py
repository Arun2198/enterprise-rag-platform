"""
Same real methodology as retrieval_relevance_guard_verification.py, but
against the actual production embedder (Jina, via a real API key) -
this is what was missing after EMBEDDING_PROVIDER=jina became the AWS
deployment default. DENSE_EMBEDDER_DEFAULT_THRESHOLD (0.68) was only
ever calibrated against BAAI/bge-small-en-v1.5's cosine-similarity
distribution; different embedding models produce different similarity
distributions, so that number was never verified to transfer to Jina -
and in production it didn't (a real, on-topic query scored 0.60,
just under threshold, triggering an incorrect abstention).

This sweeps threshold candidates and reports false positives (real
answerable queries incorrectly flagged) and true positives (genuinely
unanswerable queries correctly caught) at each, so the chosen number is
evidence-based, not guessed.

    set -a; source .env; set +a
    uv run python scripts/retrieval_relevance_guard_verification_jina.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.services.rag_service import RAGService  # noqa: E402
from rag.embeddings.jina_embedder import JinaEmbedder  # noqa: E402
from rag.guardrails.base import GuardrailContext  # noqa: E402
from rag.guardrails.retrieval_relevance_guard import RetrievalRelevanceGuard  # noqa: E402

UNANSWERABLE_QUERIES = [
    "What is the boiling point of mercury at standard atmospheric pressure?",
    "Who won the 2022 FIFA World Cup?",
    "What was NIST's total budget for fiscal year 2023?",
    "How many total lines of Python code does the AI RMF reference implementation contain?",
]

CANDIDATE_THRESHOLDS = [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68]


def main() -> int:
    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        print("JINA_API_KEY not set - nothing to verify.", file=sys.stderr)
        return 1

    embedder = JinaEmbedder(api_key=api_key)
    service = RAGService(embedder=embedder)
    service.ingest(["sample_documents/AI-RMF-1stdraft.pdf"], document_ids=["AI-RMF-1stdraft"])

    dataset = json.loads(Path("evaluation/golden_dataset.json").read_text(encoding="utf-8"))

    # Collect raw scores once (each real API call), then evaluate every
    # candidate threshold against the same scores - no need to re-embed
    # per threshold.
    answerable_scores: list[tuple[str, float]] = []
    for query in dataset["queries"]:
        retrieved = service._retrieve(query=query["query"], top_k=5)
        guard = RetrievalRelevanceGuard(embedder=service.embedder, threshold=0.0)
        context = GuardrailContext(query=query["query"], answer="x", retrieved_chunks=retrieved)
        finding = guard.check(context)
        answerable_scores.append((query["id"], finding.metadata["retrieval_relevance_score"]))

    unanswerable_scores: list[tuple[str, float]] = []
    for query in UNANSWERABLE_QUERIES:
        retrieved = service._retrieve(query=query, top_k=5)
        guard = RetrievalRelevanceGuard(embedder=service.embedder, threshold=0.0)
        context = GuardrailContext(query=query, answer="x", retrieved_chunks=retrieved)
        finding = guard.check(context)
        unanswerable_scores.append((query, finding.metadata["retrieval_relevance_score"]))

    print("=== jina-embeddings-v3 - raw scores ===")
    print(f"answerable (n={len(answerable_scores)}):")
    for qid, score in sorted(answerable_scores, key=lambda x: x[1]):
        print(f"  {score:.4f}  {qid}")
    print(f"\nunanswerable (n={len(unanswerable_scores)}):")
    for q, score in sorted(unanswerable_scores, key=lambda x: x[1]):
        print(f"  {score:.4f}  {q[:60]}")

    answerable_min = min(s for _, s in answerable_scores)
    print(f"\nanswerable minimum score: {answerable_min:.4f}")

    print("\n=== threshold sweep ===")
    for threshold in CANDIDATE_THRESHOLDS:
        false_positives = [(qid, s) for qid, s in answerable_scores if s < threshold]
        true_positives = [(q, s) for q, s in unanswerable_scores if s < threshold]
        print(
            f"threshold={threshold:.2f}  "
            f"false_positives={len(false_positives)}/{len(answerable_scores)}  "
            f"caught={len(true_positives)}/{len(unanswerable_scores)}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
