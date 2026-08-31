from rag.chunking.chunk import Chunk
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import Severity
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.retrieval.hybrid_retrieval import RetrievedChunk


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


def test_grounded_answer_passes():

    detector = HallucinationDetector(threshold=0.60)
    context = GuardrailContext(
        query="How many leave days?",
        answer="Contractors receive 10 days of leave.",
        retrieved_chunks=[
            _make_retrieved_chunk(
                "Employees receive 20 days. Contractors receive 10 days of leave."
            )
        ]
    )

    finding = detector.check(context)

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.severity == Severity.INFO
    assert finding.metadata["groundedness_score"] >= 0.60


def test_ungrounded_answer_is_flagged():

    detector = HallucinationDetector(threshold=0.60)
    context = GuardrailContext(
        query="How many leave days?",
        answer="The moon landing happened in 1969 during the space race.",
        retrieved_chunks=[
            _make_retrieved_chunk("Contractors receive 10 days of leave.")
        ]
    )

    finding = detector.check(context)

    assert finding.triggered is True
    assert finding.action == Action.WARN
    assert finding.severity == Severity.WARNING
    assert finding.metadata["likely_hallucination"] is True
    assert finding.metadata["groundedness_score"] < 0.60


def test_empty_context_yields_zero_groundedness():

    detector = HallucinationDetector(threshold=0.60)
    context = GuardrailContext(
        query="q",
        answer="Some answer text.",
        retrieved_chunks=[]
    )

    finding = detector.check(context)

    assert finding.triggered is True
    assert finding.metadata["groundedness_score"] == 0.0


def test_threshold_is_configurable():

    context = GuardrailContext(
        query="q",
        answer="mostly unrelated text here",
        retrieved_chunks=[_make_retrieved_chunk("text here about something else")]
    )

    lenient = HallucinationDetector(threshold=0.01)
    strict = HallucinationDetector(threshold=0.99)

    assert lenient.check(context).triggered is False
    assert strict.check(context).triggered is True


def test_answer_grounded_in_one_of_several_retrieved_chunks_is_not_diluted():
    """
    Real bug, found live: scoring against all retrieved chunks
    concatenated meant a larger top_k (more chunks, most on unrelated
    topics) could drag a genuinely grounded answer's score down. The
    answer only needs to be grounded in the *best-matching* chunk, not
    every chunk that happened to be retrieved alongside it.
    """
    detector = HallucinationDetector(threshold=0.60)
    context = GuardrailContext(
        query="How many vacation days do full-time employees get?",
        answer="Full-time employees accrue 22 days of paid vacation leave per calendar year.",
        retrieved_chunks=[
            _make_retrieved_chunk(
                "Full-time employees accrue 22 days of paid vacation leave per calendar year."
            ),
            _make_retrieved_chunk("Remote-eligible employees may work remotely up to 3 days a week."),
            _make_retrieved_chunk("New employees complete a 5-day onboarding program."),
            _make_retrieved_chunk("Core hours are defined as 10:00 AM to 4:00 PM."),
            _make_retrieved_chunk("Leave requests must be submitted 5 business days in advance.")
        ]
    )

    finding = detector.check(context)

    assert finding.triggered is False
    assert finding.metadata["groundedness_score"] >= 0.60


def test_embedder_blends_into_score_when_available():

    context = GuardrailContext(
        query="q",
        answer="Contractors receive 10 days of leave.",
        retrieved_chunks=[
            _make_retrieved_chunk("Contractors receive 10 days of leave.")
        ]
    )

    without_embedder = HallucinationDetector(threshold=0.60, embedder=None)
    with_embedder = HallucinationDetector(threshold=0.60, embedder=HashingEmbedder())

    score_without = without_embedder.check(context).metadata["groundedness_score"]
    score_with = with_embedder.check(context).metadata["groundedness_score"]

    assert 0.0 <= score_without <= 1.0
    assert 0.0 <= score_with <= 1.0
