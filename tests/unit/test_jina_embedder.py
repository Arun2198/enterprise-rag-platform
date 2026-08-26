from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests

from rag.embeddings.errors import EmbeddingProviderError
from rag.embeddings.jina_embedder import JinaEmbedder


def _fake_response(data, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"data": data}

    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None

    return response


@patch("rag.embeddings.jina_embedder.requests.post")
def test_embed_returns_a_single_vector(mock_post):

    mock_post.return_value = _fake_response([{"index": 0, "embedding": [0.1, 0.2, 0.3]}])
    embedder = JinaEmbedder(api_key="key")

    result = embedder.embed("hello world")

    assert result == [0.1, 0.2, 0.3]


@patch("rag.embeddings.jina_embedder.requests.post")
def test_embed_batch_sends_one_request_for_multiple_texts(mock_post):

    mock_post.return_value = _fake_response([
        {"index": 0, "embedding": [0.1]},
        {"index": 1, "embedding": [0.2]},
    ])
    embedder = JinaEmbedder(api_key="key")

    result = embedder.embed_batch(["a", "b"])

    assert result == [[0.1], [0.2]]
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["json"]["input"] == ["a", "b"]


@patch("rag.embeddings.jina_embedder.requests.post")
def test_embed_batch_reorders_results_by_index(mock_post):

    mock_post.return_value = _fake_response([
        {"index": 1, "embedding": [0.2]},
        {"index": 0, "embedding": [0.1]},
    ])
    embedder = JinaEmbedder(api_key="key")

    result = embedder.embed_batch(["a", "b"])

    assert result == [[0.1], [0.2]]


def test_embed_batch_with_empty_list_makes_no_request():

    embedder = JinaEmbedder(api_key="key")

    assert embedder.embed_batch([]) == []


@patch("rag.embeddings.jina_embedder.time.sleep")
@patch("rag.embeddings.jina_embedder.requests.post")
def test_retries_on_a_retryable_status_code_then_succeeds(mock_post, mock_sleep):

    mock_post.side_effect = [
        _fake_response([], status_code=429),
        _fake_response([{"index": 0, "embedding": [0.5]}]),
    ]
    embedder = JinaEmbedder(api_key="key", max_retries=3)

    result = embedder.embed("text")

    assert result == [0.5]
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("rag.embeddings.jina_embedder.time.sleep")
@patch("rag.embeddings.jina_embedder.requests.post")
def test_does_not_retry_a_non_retryable_status_code(mock_post, mock_sleep):

    mock_post.return_value = _fake_response([], status_code=401)
    embedder = JinaEmbedder(api_key="bad-key", max_retries=3)

    with pytest.raises(EmbeddingProviderError):
        embedder.embed("text")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


@patch("rag.embeddings.jina_embedder.time.sleep")
@patch("rag.embeddings.jina_embedder.requests.post")
def test_raises_embedding_provider_error_after_exhausting_retries(mock_post, mock_sleep):

    mock_post.return_value = _fake_response([], status_code=503)
    embedder = JinaEmbedder(api_key="key", max_retries=2)

    with pytest.raises(EmbeddingProviderError):
        embedder.embed("text")

    assert mock_post.call_count == 3


@patch("rag.embeddings.jina_embedder.requests.post")
def test_sends_the_bearer_token_and_model(mock_post):

    mock_post.return_value = _fake_response([{"index": 0, "embedding": [0.1]}])
    embedder = JinaEmbedder(api_key="secret-key", model_name="jina-embeddings-v3")

    embedder.embed("text")

    call = mock_post.call_args
    assert call.kwargs["headers"]["Authorization"] == "Bearer secret-key"
    assert call.kwargs["json"]["model"] == "jina-embeddings-v3"
