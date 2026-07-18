import time

from rag.chunking.chunk import Chunk
from rag.guardrails.manager import GuardrailManager
from rag.retrieval.hybrid_retrieval import RetrievedChunk

ITERATIONS = 2000


def _make_retrieved_chunk(text: str) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.5, keyword_score=0.5, score=0.5)


def test_guardrail_manager_output_latency_is_bounded():
    """
    Phase 1 guardrails are pure regex/token-overlap - no model inference -
    so a batch of evaluations should stay well under a second. This is a
    sanity bound against an accidental quadratic regex or O(n^2) token
    comparison, not a formal benchmark.
    """
    manager = GuardrailManager.default()
    retrieved_chunks = [
        _make_retrieved_chunk("Contractors receive 10 days of leave annually.")
    ]

    started_at = time.perf_counter()

    for _ in range(ITERATIONS):
        manager.run_output(
            query="How many leave days do contractors receive?",
            answer="Contractors receive 10 days of leave.",
            retrieved_chunks=retrieved_chunks
        )

    elapsed = time.perf_counter() - started_at
    average_latency = elapsed / ITERATIONS

    assert average_latency < 0.05


def test_guardrail_manager_throughput_scales_with_iterations():

    manager = GuardrailManager.default()
    retrieved_chunks = [_make_retrieved_chunk("Some context text.")]

    started_at = time.perf_counter()

    for _ in range(ITERATIONS):
        manager.run_output(
            query="q",
            answer="Some answer text.",
            retrieved_chunks=retrieved_chunks
        )

    elapsed = time.perf_counter() - started_at
    throughput_per_second = ITERATIONS / max(elapsed, 1e-9)

    assert throughput_per_second > 50
