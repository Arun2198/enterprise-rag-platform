from unittest.mock import patch

from rag.chunking.chunk import Chunk
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.reranker import CrossEncoderReranker


def _make_retrieved_chunk(
    chunk_id: str,
    text: str,
    score: float = 0.5
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown",
        metadata={"section": "policy"}
    )
    return RetrievedChunk(
        chunk=chunk,
        vector_score=score,
        keyword_score=score,
        score=score
    )


@patch("rag.retrieval.reranker.CrossEncoder")
def test_empty_candidate_list_returns_empty_list(mock_cross_encoder_class):

    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="anything", candidates=[], top_k=5)

    assert result == []
    mock_cross_encoder_class.return_value.predict.assert_not_called()


@patch("rag.retrieval.reranker.CrossEncoder")
def test_top_k_larger_than_candidates_returns_all_candidates(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [0.9, 0.1]
    candidates = [
        _make_retrieved_chunk("doc:0", "first chunk"),
        _make_retrieved_chunk("doc:1", "second chunk")
    ]
    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="q", candidates=candidates, top_k=10)

    assert len(result) == 2


@patch("rag.retrieval.reranker.CrossEncoder")
def test_results_sorted_by_cross_encoder_score_descending(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [0.1, 0.9, 0.5]
    candidates = [
        _make_retrieved_chunk("doc:0", "low"),
        _make_retrieved_chunk("doc:1", "high"),
        _make_retrieved_chunk("doc:2", "mid")
    ]
    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="q", candidates=candidates, top_k=3)

    assert [item.chunk.chunk_id for item in result] == ["doc:1", "doc:2", "doc:0"]
    assert result[0].score > result[1].score > result[2].score


@patch("rag.retrieval.reranker.CrossEncoder")
def test_retrieved_chunk_metadata_is_preserved(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [0.75]
    candidate = _make_retrieved_chunk("doc:0", "only chunk", score=0.4)
    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="q", candidates=[candidate], top_k=1)

    reranked = result[0]
    assert reranked.chunk.chunk_id == "doc:0"
    assert reranked.chunk.document_id == "doc"
    assert reranked.chunk.source == "doc.md"
    assert reranked.vector_score == 0.4
    assert reranked.keyword_score == 0.4
    assert reranked.chunk.metadata["section"] == "policy"
    assert reranked.chunk.metadata["cross_encoder_score"] == 0.75
    assert reranked.score == 0.75


@patch("rag.retrieval.reranker.CrossEncoder")
def test_candidate_truncation_keeps_only_top_k(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [0.9, 0.8, 0.7, 0.6]
    candidates = [
        _make_retrieved_chunk(f"doc:{i}", f"chunk {i}")
        for i in range(4)
    ]
    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="q", candidates=candidates, top_k=2)

    assert len(result) == 2
    assert [item.chunk.chunk_id for item in result] == ["doc:0", "doc:1"]


@patch("rag.retrieval.reranker.CrossEncoder")
def test_cross_encoder_receives_expected_query_chunk_pairs(mock_cross_encoder_class):

    mock_model = mock_cross_encoder_class.return_value
    mock_model.predict.return_value = [0.5, 0.5]
    candidates = [
        _make_retrieved_chunk("doc:0", "chunk text one"),
        _make_retrieved_chunk("doc:1", "chunk text two")
    ]
    reranker = CrossEncoderReranker()

    reranker.rerank(query="my query", candidates=candidates, top_k=2)

    pairs = mock_model.predict.call_args.args[0]
    assert list(pairs) == [
        ("my query", "chunk text one"),
        ("my query", "chunk text two")
    ]


@patch("rag.retrieval.reranker.CrossEncoder")
def test_duplicate_chunks_are_handled_without_error(mock_cross_encoder_class):

    mock_cross_encoder_class.return_value.predict.return_value = [0.3, 0.3]
    duplicate = _make_retrieved_chunk("doc:0", "same chunk")
    reranker = CrossEncoderReranker()

    result = reranker.rerank(query="q", candidates=[duplicate, duplicate], top_k=2)

    assert len(result) == 2
    assert all(item.chunk.chunk_id == "doc:0" for item in result)


@patch("rag.retrieval.reranker.CrossEncoder")
def test_negation_example_is_corrected(mock_cross_encoder_class):

    # a bi-encoder would likely rank the plain positive statement first;
    # the cross-encoder should catch the negation and promote the chunk
    # that actually answers the (negated) question.
    mock_cross_encoder_class.return_value.predict.return_value = [0.1, 0.95]
    candidates = [
        _make_retrieved_chunk(
            "policy:0",
            "Business class upgrades are allowed.",
            score=0.9
        ),
        _make_retrieved_chunk(
            "policy:1",
            "Business class upgrades are not allowed after check-in.",
            score=0.4
        )
    ]
    reranker = CrossEncoderReranker()

    result = reranker.rerank(
        query="Business class upgrades are not allowed.",
        candidates=candidates,
        top_k=2
    )

    assert result[0].chunk.chunk_id == "policy:1"
