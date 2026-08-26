from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests

from rag.chunking.chunk import Chunk
from rag.retrieval.errors import RerankerProviderError
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.jina_reranker import JinaReranker


def _chunk(chunk_id, text):
    return Chunk(
        chunk_id=chunk_id, document_id="doc", chunk_index=0,
        text=text, source="doc.md", document_type="markdown"
    )


def _candidate(chunk_id, text, score=0.5):
    return RetrievedChunk(chunk=_chunk(chunk_id, text), vector_score=score, keyword_score=score, score=score)


def _fake_response(results, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"results": results}

    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
    else:
        response.raise_for_status.return_value = None

    return response


def test_empty_candidate_list_returns_empty_list():

    reranker = JinaReranker(api_key="key")

    assert reranker.rerank(query="q", candidates=[], top_k=5) == []


@patch("rag.retrieval.jina_reranker.requests.post")
def test_reorders_candidates_by_relevance_score(mock_post):

    mock_post.return_value = _fake_response([
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.2},
    ])
    candidates = [_candidate("doc:0", "low"), _candidate("doc:1", "high")]
    reranker = JinaReranker(api_key="key")

    result = reranker.rerank(query="q", candidates=candidates, top_k=2)

    assert [item.chunk.chunk_id for item in result] == ["doc:1", "doc:0"]
    assert result[0].score == 0.95
    assert result[0].rank == 1
    assert result[1].rank == 2


@patch("rag.retrieval.jina_reranker.requests.post")
def test_preserves_retrieval_method_from_the_original_candidate(mock_post):

    mock_post.return_value = _fake_response([{"index": 0, "relevance_score": 0.9}])
    candidate = RetrievedChunk(
        chunk=_chunk("doc:0", "text"), vector_score=0.5, keyword_score=0.5, score=0.5,
        retrieval_method="both"
    )
    reranker = JinaReranker(api_key="key")

    result = reranker.rerank(query="q", candidates=[candidate], top_k=1)

    assert result[0].retrieval_method == "both"


@patch("rag.retrieval.jina_reranker.requests.post")
def test_sends_top_n_capped_to_candidate_count(mock_post):

    mock_post.return_value = _fake_response([{"index": 0, "relevance_score": 0.9}])
    reranker = JinaReranker(api_key="key")

    reranker.rerank(query="q", candidates=[_candidate("doc:0", "text")], top_k=50)

    assert mock_post.call_args.kwargs["json"]["top_n"] == 1


@patch("rag.retrieval.jina_reranker.time.sleep")
@patch("rag.retrieval.jina_reranker.requests.post")
def test_raises_after_exhausting_retries(mock_post, mock_sleep):

    mock_post.return_value = _fake_response([], status_code=503)
    reranker = JinaReranker(api_key="key", max_retries=2)

    with pytest.raises(RerankerProviderError):
        reranker.rerank(query="q", candidates=[_candidate("doc:0", "text")], top_k=1)

    assert mock_post.call_count == 3
