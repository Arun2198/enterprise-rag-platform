import json
from unittest.mock import MagicMock
from unittest.mock import patch

from rag.chunking.chunk import Chunk
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.llm_judge_hallucination_detector import LLMJudgeHallucinationDetector
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class FakeAPIError(Exception):

    def __init__(self, status_code: int) -> None:
        super().__init__(f"api error {status_code}")
        self.status_code = status_code


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


def _make_completion(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


def _make_context(answer: str, context_text: str) -> GuardrailContext:
    return GuardrailContext(
        query="q",
        answer=answer,
        retrieved_chunks=[_make_retrieved_chunk(context_text)]
    )


@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_grounded_verdict_passes(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion(
        json.dumps({"groundedness_score": 0.95, "reasoning": "fully supported"})
    )
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        threshold=0.6
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.metadata["groundedness_score"] == 0.95
    assert finding.metadata["judge_available"] is True


@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_ungrounded_verdict_is_flagged(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion(
        json.dumps({"groundedness_score": 0.1, "reasoning": "not supported"})
    )
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        threshold=0.6
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.triggered is True
    assert finding.action == Action.WARN
    assert finding.metadata["groundedness_score"] == 0.1


@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_verdict_wrapped_in_markdown_fence_is_parsed(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion(
        '```json\n{"groundedness_score": 0.8, "reasoning": "mostly supported"}\n```'
    )
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        threshold=0.6
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.triggered is False
    assert finding.metadata["groundedness_score"] == 0.8


@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_unparseable_response_fails_open(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion("not json at all")
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.metadata["judge_available"] is False


@patch("time.sleep", return_value=None)
@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_api_failure_fails_open_without_crashing(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = FakeAPIError(status_code=401)
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.triggered is False
    assert finding.metadata["judge_available"] is False
    assert mock_client.chat.completions.create.call_count == 1


@patch("time.sleep", return_value=None)
@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_retries_only_transient_failures(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = [
        FakeAPIError(status_code=503),
        _make_completion(json.dumps({"groundedness_score": 0.9, "reasoning": "ok"}))
    ]
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    finding = detector.check(_make_context("answer", "context"))

    assert finding.metadata["judge_available"] is True
    assert mock_client.chat.completions.create.call_count == 2
    assert mock_sleep.call_count == 1


@patch("rag.guardrails.llm_judge_hallucination_detector.OpenAI")
def test_empty_context_does_not_call_judge(mock_openai_class):

    mock_client = mock_openai_class.return_value
    detector = LLMJudgeHallucinationDetector(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    finding = detector.check(GuardrailContext(query="q", answer="answer", retrieved_chunks=[]))

    assert finding.triggered is False
    assert finding.metadata["judge_available"] is False
    mock_client.chat.completions.create.assert_not_called()
