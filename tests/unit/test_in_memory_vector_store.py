from rag.chunking.chunk import Chunk
from rag.vector_store.in_memory_store import InMemoryVectorStore


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        chunk_index=0,
        source="doc.md",
        document_type="markdown",
        text="some text"
    )


def test_get_returns_the_chunk_for_a_known_id():

    store = InMemoryVectorStore()
    chunk = _chunk("doc:0")
    store.add(chunk, [1.0, 0.0])

    assert store.get("doc:0") is chunk


def test_get_returns_none_for_an_unknown_id():

    store = InMemoryVectorStore()

    assert store.get("does-not-exist") is None
