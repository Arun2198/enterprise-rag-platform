from rag.chunking.chunk import Chunk
from rag.vector_store.in_memory_store import InMemoryVectorStore


def _chunk(chunk_id: str, document_id: str = "doc", text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        source="doc.md",
        document_type="markdown",
        text=text,
        metadata={"document_id": document_id}
    )


def test_get_returns_the_chunk_for_a_known_id():

    store = InMemoryVectorStore()
    chunk = _chunk("doc:0")
    store.add(chunk, [1.0, 0.0])

    assert store.get("doc:0") is chunk


def test_get_returns_none_for_an_unknown_id():

    store = InMemoryVectorStore()

    assert store.get("does-not-exist") is None


def test_get_embedding_returns_the_stored_vector():

    store = InMemoryVectorStore()
    store.add(_chunk("doc:0"), [1.0, 0.0])

    assert store.get_embedding("doc:0") == [1.0, 0.0]


def test_get_embedding_returns_none_for_an_unknown_id():

    store = InMemoryVectorStore()

    assert store.get_embedding("does-not-exist") is None


def test_search_metadata_filter_scopes_to_a_single_document():
    store = InMemoryVectorStore()
    store.add(_chunk("doc-a:0", document_id="doc-a"), [1.0, 0.0])
    store.add(_chunk("doc-b:0", document_id="doc-b"), [1.0, 0.0])

    results = store.search([1.0, 0.0], top_k=5, metadata_filter={"document_id": "doc-a"})

    assert [r.chunk.chunk_id for r in results] == ["doc-a:0"]


def test_search_metadata_filter_accepts_a_list_for_in_semantics():
    store = InMemoryVectorStore()
    store.add(_chunk("doc-a:0", document_id="doc-a"), [1.0, 0.0])
    store.add(_chunk("doc-b:0", document_id="doc-b"), [1.0, 0.0])
    store.add(_chunk("doc-c:0", document_id="doc-c"), [1.0, 0.0])

    results = store.search([1.0, 0.0], top_k=5, metadata_filter={"document_id": ["doc-a", "doc-c"]})

    assert {r.chunk.chunk_id for r in results} == {"doc-a:0", "doc-c:0"}


def test_search_lexical_metadata_filter_accepts_a_list_for_in_semantics():
    store = InMemoryVectorStore()
    store.add(_chunk("doc-a:0", document_id="doc-a", text="contractors receive leave"), [1.0])
    store.add(_chunk("doc-b:0", document_id="doc-b", text="contractors receive leave"), [1.0])
    store.add(_chunk("doc-c:0", document_id="doc-c", text="contractors receive leave"), [1.0])

    results = store.search_lexical(
        "contractors receive leave", top_k=5, metadata_filter={"document_id": ["doc-a", "doc-c"]}
    )

    assert {r.chunk.chunk_id for r in results} == {"doc-a:0", "doc-c:0"}


def test_search_filter_is_applied_before_top_k_truncation_not_after():
    """
    With top_k=1 and the higher-scoring chunk excluded by the filter,
    the lower-scoring but in-scope chunk must still win the single slot -
    proves filtering narrows the candidate set before ranking/truncation,
    not after (a post-hoc filter would leave this empty).
    """
    store = InMemoryVectorStore()
    store.add(_chunk("doc-a:0", document_id="doc-a"), [1.0, 0.0])  # perfect match, excluded
    store.add(_chunk("doc-b:0", document_id="doc-b"), [0.1, 0.99])  # weak match, in scope

    results = store.search([1.0, 0.0], top_k=1, metadata_filter={"document_id": "doc-b"})

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc-b:0"
