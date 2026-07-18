from unittest.mock import patch

from rag.chunking.chunk import Chunk
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.nli_hallucination_detector import NLIHallucinationDetector
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


@patch("rag.guardrails.nli_hallucination_detector.CrossEncoder")
def test_high_entailment_probability_passes(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [[0.01, 0.95, 0.04]]
    detector = NLIHallucinationDetector(threshold=0.5)
    context = GuardrailContext(
        query="q",
        answer="Contractors receive 10 days of leave.",
        retrieved_chunks=[_make_retrieved_chunk("Contractors receive 10 days of leave.")]
    )

    finding = detector.check(context)

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.metadata["groundedness_score"] == 0.95


@patch("rag.guardrails.nli_hallucination_detector.CrossEncoder")
def test_low_entailment_probability_is_flagged(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [[0.9, 0.05, 0.05]]
    detector = NLIHallucinationDetector(threshold=0.5)
    context = GuardrailContext(
        query="q",
        answer="The moon landing happened in 1969.",
        retrieved_chunks=[_make_retrieved_chunk("Contractors receive 10 days of leave.")]
    )

    finding = detector.check(context)

    assert finding.triggered is True
    assert finding.action == Action.WARN
    assert finding.metadata["groundedness_score"] == 0.05


@patch("rag.guardrails.nli_hallucination_detector.CrossEncoder")
def test_max_entailment_across_multiple_chunks_is_used(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [
        [0.9, 0.05, 0.05],
        [0.02, 0.9, 0.08],
        [0.5, 0.3, 0.2]
    ]
    detector = NLIHallucinationDetector(threshold=0.5)
    context = GuardrailContext(
        query="q",
        answer="answer",
        retrieved_chunks=[
            _make_retrieved_chunk("unrelated chunk one"),
            _make_retrieved_chunk("the chunk that actually supports the answer"),
            _make_retrieved_chunk("somewhat related chunk")
        ]
    )

    finding = detector.check(context)

    assert finding.triggered is False
    assert finding.metadata["groundedness_score"] == 0.9


@patch("rag.guardrails.nli_hallucination_detector.CrossEncoder")
def test_empty_context_is_flagged_without_calling_model(mock_cross_encoder_class):

    detector = NLIHallucinationDetector(threshold=0.5)
    context = GuardrailContext(query="q", answer="answer", retrieved_chunks=[])

    finding = detector.check(context)

    assert finding.triggered is True
    assert finding.metadata["groundedness_score"] == 0.0
    mock_cross_encoder_class.return_value.predict.assert_not_called()


@patch("rag.guardrails.nli_hallucination_detector.CrossEncoder")
def test_premise_hypothesis_order_is_chunk_then_answer(mock_cross_encoder_class):

    mock_model = mock_cross_encoder_class.return_value
    mock_model.predict.return_value = [[0.0, 1.0, 0.0]]
    detector = NLIHallucinationDetector()
    context = GuardrailContext(
        query="q",
        answer="the answer text",
        retrieved_chunks=[_make_retrieved_chunk("the chunk text")]
    )

    detector.check(context)

    pairs = mock_model.predict.call_args.args[0]
    assert pairs == [("the chunk text", "the answer text")]
