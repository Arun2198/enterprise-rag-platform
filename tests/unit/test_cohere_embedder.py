from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests

from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.embeddings.errors import EmbeddingProviderError


def _fake_response(embeddings, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"embeddings": {"float": embeddings}}

    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None

    return response


@patch("rag.embeddings.cohere_embedder.requests.post")
def test_embed_returns_a_single_vector(mock_post):

    mock_post.return_value = _fake_response([[0.1, 0.2, 0.3]])
    embedder = CohereEmbedder(api_key="key")

    result = embedder.embed("hello world")

    assert result == [0.1, 0.2, 0.3]


@patch("rag.embeddings.cohere_embedder.requests.post")
def test_embed_batch_sends_one_request_for_multiple_texts(mock_post):

    mock_post.return_value = _fake_response([[0.1], [0.2]])
    embedder = CohereEmbedder(api_key="key")

    result = embedder.embed_batch(["a", "b"])

    assert result == [[0.1], [0.2]]
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["json"]["texts"] == ["a", "b"]


def test_embed_batch_with_empty_list_makes_no_request():

    embedder = CohereEmbedder(api_key="key")

    assert embedder.embed_batch([]) == []


@patch("rag.embeddings.cohere_embedder.time.sleep")
@patch("rag.embeddings.cohere_embedder.requests.post")
def test_retries_on_a_retryable_status_code_then_succeeds(mock_post, mock_sleep):

    mock_post.side_effect = [
        _fake_response([], status_code=500),
        _fake_response([[0.5]]),
    ]
    embedder = CohereEmbedder(api_key="key", max_retries=3)

    result = embedder.embed("text")

    assert result == [0.5]
    assert mock_post.call_count == 2


@patch("rag.embeddings.cohere_embedder.time.sleep")
@patch("rag.embeddings.cohere_embedder.requests.post")
def test_does_not_retry_a_non_retryable_status_code(mock_post, mock_sleep):

    mock_post.return_value = _fake_response([], status_code=401)
    embedder = CohereEmbedder(api_key="bad-key", max_retries=3)

    with pytest.raises(EmbeddingProviderError):
        embedder.embed("text")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


@patch("rag.embeddings.cohere_embedder.requests.post")
def test_sends_the_bearer_token_input_type_and_model(mock_post):

    mock_post.return_value = _fake_response([[0.1]])
    embedder = CohereEmbedder(api_key="secret-key", model_name="embed-english-v3.0")

    embedder.embed("text")

    call = mock_post.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert call.kwargs["json"]["model"] == "embed-english-v3.0"
    assert call.kwargs["json"]["input_type"] == "search_document"
