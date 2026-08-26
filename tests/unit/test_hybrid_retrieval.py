from rag.chunking.chunk import Chunk
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.vector_store.in_memory_store import SearchResult


class _StubEmbedder:

    dimensions = 2

    def embed(self, text):
        return [1.0, 0.0]


class _StubVectorStore:

    def __init__(self, dense_results=None, lexical_results=None):
        self.dense_results = dense_results or []
        self.lexical_results = lexical_results or []
        self.search_calls = []
        self.lexical_calls = []

    def search(self, query_embedding, top_k=5, metadata_filter=None):
        self.search_calls.append({"top_k": top_k, "metadata_filter": metadata_filter})
        return self.dense_results[:top_k]

    def search_lexical(self, query_text, top_k=5, metadata_filter=None):
        self.lexical_calls.append({"top_k": top_k, "metadata_filter": metadata_filter})
        return self.lexical_results[:top_k]


def _chunk(chunk_id: str, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )


def test_a_chunk_found_by_both_methods_outranks_one_found_by_only_one():

    store = _StubVectorStore(
        dense_results=[
            SearchResult(chunk=_chunk("doc:0"), score=0.9),
            SearchResult(chunk=_chunk("doc:1"), score=0.5),
        ],
        lexical_results=[
            SearchResult(chunk=_chunk("doc:1"), score=5.0),
        ]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("query", top_k=5)

    # doc:1 - rank 2 in dense (1/62), rank 1 in lexical (1/61) -> higher RRF
    # doc:0 - rank 1 in dense only (1/61)
    assert results[0].chunk.chunk_id == "doc:1"
    assert results[0].retrieval_method == "both"
    assert results[1].chunk.chunk_id == "doc:0"
    assert results[1].retrieval_method == "dense"


def test_rrf_score_matches_the_formula():

    store = _StubVectorStore(
        dense_results=[SearchResult(chunk=_chunk("doc:0"), score=0.9)],
        lexical_results=[]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder(), rrf_k=60)

    results = retriever.retrieve("query", top_k=5)

    assert results[0].score == 1.0 / (60 + 1)


def test_retrieval_method_dense_only():

    store = _StubVectorStore(
        dense_results=[SearchResult(chunk=_chunk("doc:0"), score=0.9)],
        lexical_results=[]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("query", top_k=5)

    assert results[0].retrieval_method == "dense"
    assert results[0].vector_score == 0.9
    assert results[0].keyword_score == 0.0


def test_retrieval_method_bm25_only():

    store = _StubVectorStore(
        dense_results=[],
        lexical_results=[SearchResult(chunk=_chunk("doc:0"), score=3.2)]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("query", top_k=5)

    assert results[0].retrieval_method == "bm25"
    assert results[0].keyword_score == 3.2
    assert results[0].vector_score == 0.0


def test_rank_is_one_indexed_and_sequential():

    store = _StubVectorStore(
        dense_results=[
            SearchResult(chunk=_chunk(f"doc:{i}"), score=1.0 - i * 0.1)
            for i in range(4)
        ]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("query", top_k=4)

    assert [r.rank for r in results] == [1, 2, 3, 4]


def test_results_are_truncated_to_top_k():

    store = _StubVectorStore(
        dense_results=[
            SearchResult(chunk=_chunk(f"doc:{i}"), score=1.0 - i * 0.1)
            for i in range(10)
        ]
    )
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    results = retriever.retrieve("query", top_k=3)

    assert len(results) == 3


def test_queries_dense_and_lexical_at_their_own_configured_depths():

    store = _StubVectorStore()
    retriever = HybridRetriever(
        vector_store=store,
        embedder=_StubEmbedder(),
        dense_top_k=15,
        bm25_top_k=25
    )

    retriever.retrieve("query", top_k=5)

    assert store.search_calls[0]["top_k"] == 15
    assert store.lexical_calls[0]["top_k"] == 25


def test_metadata_filter_is_passed_to_both_searches():

    store = _StubVectorStore()
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    retriever.retrieve("query", top_k=5, metadata_filter={"doc_type": "policy"})

    assert store.search_calls[0]["metadata_filter"] == {"doc_type": "policy"}
    assert store.lexical_calls[0]["metadata_filter"] == {"doc_type": "policy"}


def test_returns_empty_list_when_neither_method_has_results():

    store = _StubVectorStore()
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder())

    assert retriever.retrieve("query", top_k=5) == []


def test_custom_rrf_k_changes_the_fused_score():

    store = _StubVectorStore(dense_results=[SearchResult(chunk=_chunk("doc:0"), score=0.9)])
    retriever = HybridRetriever(vector_store=store, embedder=_StubEmbedder(), rrf_k=10)

    results = retriever.retrieve("query", top_k=5)

    assert results[0].score == 1.0 / (10 + 1)
