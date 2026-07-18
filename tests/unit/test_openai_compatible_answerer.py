from unittest.mock import MagicMock
from unittest.mock import patch

from rag.chunking.chunk import Chunk
from rag.generation.openai_compatible_answerer import FALLBACK_ERROR
from rag.generation.openai_compatible_answerer import FALLBACK_NO_CONTEXT
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class FakeAPIError(Exception):

    def __init__(self, status_code: int) -> None:
        super().__init__(f"api error {status_code}")
        self.status_code = status_code


def _make_retrieved_chunk(
    chunk_id: str = "doc:0",
    text: str = "Only this context may be used."
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )
    return RetrievedChunk(
        chunk=chunk,
        vector_score=0.8,
        keyword_score=0.5,
        score=0.7
    )


def _make_completion(content: str = "Grounded answer.") -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_prompt_contains_retrieved_chunks_and_source_ids(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion()
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    answerer.answer("What can be used?", [_make_retrieved_chunk()])

    prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]

    assert "Only this context may be used." in prompt
    assert "doc:0" in prompt


@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_empty_retrieval_returns_fallback_without_calling_client(mock_openai_class):

    mock_client = mock_openai_class.return_value
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    answer = answerer.answer("What can be used?", [])

    assert answer == FALLBACK_NO_CONTEXT
    mock_client.chat.completions.create.assert_not_called()


@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_configured_model_name_is_passed_to_client(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion()
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="custom-model"
    )

    answerer.answer("question", [_make_retrieved_chunk()])

    assert mock_client.chat.completions.create.call_args.kwargs["model"] == "custom-model"


@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_configured_timeout_is_used(mock_openai_class):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion()
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        timeout=12.5
    )

    answerer.answer("question", [_make_retrieved_chunk()])

    assert mock_openai_class.call_args.kwargs["timeout"] == 12.5
    assert mock_client.chat.completions.create.call_args.kwargs["timeout"] == 12.5


@patch("time.sleep", return_value=None)
@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_api_exception_returns_fallback_message(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = FakeAPIError(status_code=401)
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    )

    answer = answerer.answer("question", [_make_retrieved_chunk()])

    assert answer == FALLBACK_ERROR
    assert mock_client.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_retry_logic_retries_only_transient_failures(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = [
        FakeAPIError(status_code=503),
        FakeAPIError(status_code=503),
        _make_completion("Recovered answer.")
    ]
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        max_retries=3
    )

    answer = answerer.answer("question", [_make_retrieved_chunk()])

    assert answer == "Recovered answer."
    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_non_transient_failure_is_not_retried(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = FakeAPIError(status_code=400)
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        max_retries=3
    )

    answer = answerer.answer("question", [_make_retrieved_chunk()])

    assert answer == FALLBACK_ERROR
    assert mock_client.chat.completions.create.call_count == 1
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_retries_exhausted_returns_fallback_message(mock_openai_class, mock_sleep):

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.side_effect = FakeAPIError(status_code=503)
    answerer = OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini",
        max_retries=2
    )

    answer = answerer.answer("question", [_make_retrieved_chunk()])

    assert answer == FALLBACK_ERROR
    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2
