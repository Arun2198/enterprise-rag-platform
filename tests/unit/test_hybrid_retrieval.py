import math

from rag.chunking.chunk import Chunk
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.vector_store.in_memory_store import SearchResult


class _StubEmbedder:

    def embed(self, text):
        return [1.0, 0.0]


class _StubVectorStore:

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query_embedding, top_k=5, metadata_filter=None):
        self.calls.append({"top_k": top_k, "metadata_filter": metadata_filter})
        return self.results[:top_k]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )


def test_fuses_vector_and_keyword_scores_with_default_weights():

    store = _StubVectorStore([
        SearchResult(chunk=_chunk("doc:0", "contractors receive leave"), score=0.8)
    ])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("contractors leave", top_k=5)

    # keyword_score: full overlap (2/2 tokens) -> log1p(2)/log1p(2) = 1.0
    # fused: 0.65 * 0.8 + 0.35 * 1.0
    expected = 0.65 * 0.8 + 0.35 * 1.0
    assert math.isclose(results[0].vector_score, 0.8)
    assert math.isclose(results[0].keyword_score, 1.0)
    assert math.isclose(results[0].score, expected)


def test_custom_weights_change_the_fused_score():

    store = _StubVectorStore([
        SearchResult(chunk=_chunk("doc:0", "contractors receive leave"), score=0.8)
    ])
    retriever = HybridRetriever(
        vector_store=store,
        embedder=_StubEmbedder(),
        vector_weight=0.9,
        keyword_weight=0.1
    )

    results = retriever.retrieve("contractors leave", top_k=5)

    expected = 0.9 * 0.8 + 0.1 * 1.0
    assert math.isclose(results[0].score, expected)


def test_partial_keyword_overlap_scores_between_zero_and_one():

    store = _StubVectorStore([
        SearchResult(chunk=_chunk("doc:0", "contractors receive something else"), score=0.5)
    ])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("contractors leave policy", top_k=5)

    # 1 of 3 query terms overlaps ("contractors")
    expected_keyword_score = math.log1p(1) / math.log1p(3)
    assert math.isclose(results[0].keyword_score, expected_keyword_score)


def test_keyword_score_is_zero_when_query_has_no_alphanumeric_tokens():

    store = _StubVectorStore([
        SearchResult(chunk=_chunk("doc:0", "some content here"), score=0.5)
    ])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("!!!", top_k=5)

    assert results[0].keyword_score == 0.0


def test_results_are_sorted_by_fused_score_descending():

    store = _StubVectorStore([
        # 0.65 * 0.5 + 0.35 * 0.0 = 0.325
        SearchResult(chunk=_chunk("doc:0", "irrelevant text here"), score=0.5),
        # 0.65 * 0.3 + 0.35 * 1.0 = 0.545
        SearchResult(chunk=_chunk("doc:1", "contractors receive leave"), score=0.3),
    ])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("contractors leave", top_k=5)

    assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)
    # the lower-vector-score-but-full-keyword-overlap chunk still wins
    # under the default 0.65/0.35 weighting
    assert results[0].chunk.chunk_id == "doc:1"


def test_results_are_truncated_to_top_k():

    store = _StubVectorStore([
        SearchResult(chunk=_chunk(f"doc:{i}", "contractors leave"), score=1.0 - i * 0.1)
        for i in range(10)
    ])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("contractors leave", top_k=3)

    assert len(results) == 3


def test_over_fetches_from_the_vector_store_by_4x_top_k():

    store = _StubVectorStore([])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    retriever.retrieve("query", top_k=5)

    assert store.calls[0]["top_k"] == 20


def test_metadata_filter_is_passed_through_to_the_vector_store():

    store = _StubVectorStore([])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    retriever.retrieve("query", top_k=5, metadata_filter={"doc_type": "policy"})

    assert store.calls[0]["metadata_filter"] == {"doc_type": "policy"}


def test_returns_empty_list_when_vector_store_has_no_results():

    store = _StubVectorStore([])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    assert retriever.retrieve("query", top_k=5) == []
