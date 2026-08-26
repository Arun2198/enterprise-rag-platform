"""
Real end-to-end verification for RetrievalRelevanceGuard against a genuine
dense embedder (BAAI/bge-small-en-v1.5 via SentenceTransformerEmbedder,
not the tests/unit/conftest.py fake that delegates to HashingEmbedder for
speed). This is the evidence behind DENSE_EMBEDDER_DEFAULT_THRESHOLD and
HASHING_EMBEDDER_DEFAULT_THRESHOLD in
rag/guardrails/retrieval_relevance_guard.py - run this again if
EMBEDDING_MODEL_NAME changes to a materially different model, or if
evaluation/golden_dataset.json's queries change.

    uv run python scripts/retrieval_relevance_guard_verification.py

Downloads a real model from HuggingFace on first run - not part of the
fast/offline pytest suite by design.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.services.rag_service import RAGService  # noqa: E402
from rag.embeddings.hashing_embedder import HashingEmbedder  # noqa: E402
from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder  # noqa: E402
from rag.guardrails.base import GuardrailContext  # noqa: E402
from rag.guardrails.retrieval_relevance_guard import DENSE_EMBEDDER_DEFAULT_THRESHOLD  # noqa: E402
from rag.guardrails.retrieval_relevance_guard import HASHING_EMBEDDER_DEFAULT_THRESHOLD  # noqa: E402
from rag.guardrails.retrieval_relevance_guard import RetrievalRelevanceGuard  # noqa: E402

UNANSWERABLE_QUERIES = [
    "What is the boiling point of mercury at standard atmospheric pressure?",
    "Who won the 2022 FIFA World Cup?",
    "What was NIST's total budget for fiscal year 2023?",
    "How many total lines of Python code does the AI RMF reference implementation contain?",
]


def run(
    label: str,
    embedder,
    threshold: float
) -> None:
    service = RAGService(embedder=embedder)
    service.ingest(["sample_documents/AI-RMF-1stdraft.pdf"])
    guard = RetrievalRelevanceGuard(embedder=service.embedder, threshold=threshold)

    dataset = json.loads(Path("evaluation/golden_dataset.json").read_text(encoding="utf-8"))
    false_positives = []

    for query in dataset["queries"]:
        retrieved = service._retrieve(query=query["query"], top_k=5)
        context = GuardrailContext(query=query["query"], answer="x", retrieved_chunks=retrieved)
        finding = guard.check(context)

        if finding.triggered:
            false_positives.append((query["id"], finding.metadata["retrieval_relevance_score"]))

    true_positives = []

    for query in UNANSWERABLE_QUERIES:
        retrieved = service._retrieve(query=query, top_k=5)
        context = GuardrailContext(query=query, answer="x", retrieved_chunks=retrieved)
        finding = guard.check(context)

        if finding.triggered:
            true_positives.append((query, finding.metadata["retrieval_relevance_score"]))

    print(f"=== {label} (threshold={threshold}) ===")
    print(f"false positives on {len(dataset['queries'])} real answerable golden queries: "
          f"{len(false_positives)} {false_positives}")
    print(f"caught {len(true_positives)}/{len(UNANSWERABLE_QUERIES)} known-gap unanswerable "
          f"queries: {[q[:40] for q, _ in true_positives]}")


if __name__ == "__main__":
    run("hashing", HashingEmbedder(), HASHING_EMBEDDER_DEFAULT_THRESHOLD)
    run("sentence_transformer (bge-small)", SentenceTransformerEmbedder(), DENSE_EMBEDDER_DEFAULT_THRESHOLD)
