from ingestion.contracts.document import Document
from ingestion.contracts.manifest import ChunkStatus
from ingestion.incremental_indexer import IncrementalIndexer
from ingestion.manifest_store import InMemoryManifestStore
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.vector_store.in_memory_store import InMemoryVectorStore


class CountingEmbedder:
    """
    Wraps HashingEmbedder and counts exactly how many texts were actually
    sent for embedding across all embed_batch calls - what Phase 19's
    "assert the exact number of embedding calls" requirement means in
    practice, regardless of how many batch calls that's split across.
    """

    def __init__(self, dimensions: int = 384, model_name: str = "hashing-384"):
        self._inner = HashingEmbedder(dimensions=dimensions)
        self.dimensions = dimensions
        self.provider_name = "hashing"
        self.model_name = model_name
        self.embedded_texts: list[str] = []
        self.batch_calls = 0

    def embed(self, text: str) -> list[float]:
        self.embedded_texts.append(text)
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.embedded_texts.extend(texts)
        return self._inner.embed_batch(texts)

    @property
    def call_count(self) -> int:
        return len(self.embedded_texts)


class FlakyEmbedder(CountingEmbedder):
    """Fails the Nth embed_batch call (1-indexed), succeeds otherwise -
    for exercising partial-failure/retry behavior deterministically."""

    def __init__(self, fail_on_call: int, **kwargs):
        super().__init__(**kwargs)
        self.fail_on_call = fail_on_call

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        if self.batch_calls == self.fail_on_call:
            raise RuntimeError("simulated embedding provider outage")
        self.embedded_texts.extend(texts)
        return self._inner.embed_batch(texts)


def _page(n: int) -> str:
    return (
        f"Page {n} Heading\n"
        f"This is the real body content of page {n}, describing topic "
        f"number {n} of the document in enough distinct words to form "
        f"its own chunk."
    )


def _make_document(document_id: str, page_count: int, changed_pages: dict[int, str] | None = None) -> Document:
    changed_pages = changed_pages or {}
    pages = [changed_pages.get(n, _page(n)) for n in range(1, page_count + 1)]
    return Document(
        document_id=document_id,
        source=f"{document_id}.pdf",
        document_type="pdf",
        content="\n".join(pages),
        pages=pages,
        metadata={}
    )


def _build_indexer(embedder=None):
    embedder = embedder or CountingEmbedder()
    vector_store = InMemoryVectorStore()
    manifest_store = InMemoryManifestStore()
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    indexer = IncrementalIndexer(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        manifest_store=manifest_store
    )
    return indexer, embedder, vector_store, manifest_store


# 1. new document -> all chunks embedded
def test_new_document_embeds_every_chunk():
    indexer, embedder, vector_store, _ = _build_indexer()
    doc = _make_document("doc-a", page_count=5)

    result = indexer.index(doc)

    assert result.previous_version is None
    assert result.new_version == 1
    assert result.changed_chunks == result.total_chunks
    assert result.unchanged_chunks == 0
    assert result.embedding_calls == result.total_chunks
    assert embedder.call_count == result.total_chunks
    assert vector_store.count() == result.total_chunks


# 2 / 9. identical re-upload -> zero new embeddings, idempotent
def test_identical_reupload_triggers_zero_embeddings():
    indexer, embedder, vector_store, _ = _build_indexer()
    doc = _make_document("doc-b", page_count=5)
    indexer.index(doc)
    first_call_count = embedder.call_count
    first_vector_count = vector_store.count()

    result = indexer.index(doc)

    assert result.changed_chunks == 0
    assert result.embedding_calls == 0
    assert embedder.call_count == first_call_count
    assert vector_store.count() == first_vector_count
    assert result.new_version == 1  # content unchanged - no version bump


# 3. one page changed in a multi-page document -> only that page's chunks re-embedded
def test_single_page_change_only_reembeds_that_pages_chunks():
    indexer, embedder, vector_store, _ = _build_indexer()
    doc = _make_document("doc-c", page_count=10)
    indexer.index(doc)
    embedder.embedded_texts.clear()

    changed = _make_document("doc-c", page_count=10, changed_pages={
        6: "Page 6 Heading\nCompletely rewritten content for page six only."
    })
    result = indexer.index(changed)

    assert result.changed_pages == 1
    assert result.unchanged_pages == 9
    assert result.changed_chunks == 1
    assert result.unchanged_chunks == result.total_chunks - 1
    assert result.embedding_calls == 1
    assert embedder.call_count == 1
    assert "page six only" in embedder.embedded_texts[0]
    assert result.new_version == 2


# 4. two pages changed -> only those pages' chunks re-embedded
def test_two_page_change_only_reembeds_those_pages_chunks():
    indexer, embedder, vector_store, _ = _build_indexer()
    doc = _make_document("doc-d", page_count=100)
    indexer.index(doc)
    embedder.embedded_texts.clear()

    changed = _make_document("doc-d", page_count=100, changed_pages={
        37: "Page 37 Heading\nRewritten content for page thirty-seven.",
        38: "Page 38 Heading\nRewritten content for page thirty-eight.",
    })
    result = indexer.index(changed)

    assert result.changed_pages == 2
    assert result.unchanged_pages == 98
    assert result.changed_chunks == 2
    assert result.embedding_calls == 2
    assert embedder.call_count == 2


# 5. page deleted -> its old chunks removed from the vector store
def test_deleted_page_removes_its_chunks():
    indexer, embedder, vector_store, manifest_store = _build_indexer()
    doc = _make_document("doc-e", page_count=5)
    indexer.index(doc)
    manifest_before = manifest_store.get("doc-e")
    page5_chunk_ids = next(p.chunk_ids for p in manifest_before.pages if p.page_number == 5)
    assert page5_chunk_ids
    for chunk_id in page5_chunk_ids:
        assert vector_store.get(chunk_id) is not None

    shorter = _make_document("doc-e", page_count=4)
    result = indexer.index(shorter)

    assert result.deleted_pages == 1
    assert result.deleted_chunks == len(page5_chunk_ids)
    for chunk_id in page5_chunk_ids:
        assert vector_store.get(chunk_id) is None


# 6. document deleted -> its vectors no longer retrievable, manifest cleared
def test_document_delete_removes_all_chunks_and_manifest():
    from app.services.rag_service import RAGService

    manifest_store = InMemoryManifestStore()
    embedder = CountingEmbedder()
    vector_store = InMemoryVectorStore()
    service = RAGService(
        embedder=embedder,
        vector_store=vector_store,
        manifest_store=manifest_store,
        chunker=RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    )
    doc = _make_document("doc-f", page_count=3)
    service.index_document(doc)
    assert vector_store.count() > 0
    assert manifest_store.get("doc-f") is not None

    deleted = service.delete_document("doc-f")

    assert deleted > 0
    assert vector_store.count() == 0
    assert manifest_store.get("doc-f") is None


# 7. per-page chunking means no chunk can span two pages by construction -
# editing one page never touches a neighboring page's chunks, even at the
# boundary between them.
def test_chunks_never_span_pages_and_edit_does_not_touch_neighbor():
    indexer, embedder, vector_store, manifest_store = _build_indexer()
    doc = _make_document("doc-g", page_count=3)
    indexer.index(doc)
    manifest = manifest_store.get("doc-g")

    page_chunk_ids = {p.page_number: set(p.chunk_ids) for p in manifest.pages}
    assert page_chunk_ids[1].isdisjoint(page_chunk_ids[2])
    assert page_chunk_ids[2].isdisjoint(page_chunk_ids[3])

    page2_chunk_ids_before = page_chunk_ids[2]

    changed = _make_document("doc-g", page_count=3, changed_pages={
        2: "Page 2 Heading\nEntirely rewritten page two content, right up to the boundary."
    })
    result = indexer.index(changed)
    manifest_after = manifest_store.get("doc-g")
    page1_chunk_ids_after = next(p.chunk_ids for p in manifest_after.pages if p.page_number == 1)
    page3_chunk_ids_after = next(p.chunk_ids for p in manifest_after.pages if p.page_number == 3)

    assert result.changed_pages == 1
    assert set(page1_chunk_ids_after) == page_chunk_ids[1]
    assert set(page3_chunk_ids_after) == page_chunk_ids[3]
    assert page2_chunk_ids_before  # sanity: page 2 actually had chunks


# 8. embedding model changed, identical content -> full re-embed regardless of hash match
def test_embedding_fingerprint_change_forces_full_reembed():
    manifest_store = InMemoryManifestStore()
    vector_store = InMemoryVectorStore()
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    doc = _make_document("doc-h", page_count=4)

    embedder_v1 = CountingEmbedder(dimensions=384, model_name="model-v1")
    IncrementalIndexer(chunker, embedder_v1, vector_store, manifest_store).index(doc)

    embedder_v2 = CountingEmbedder(dimensions=768, model_name="model-v2")
    result = IncrementalIndexer(chunker, embedder_v2, vector_store, manifest_store).index(doc)

    assert result.changed_chunks == result.total_chunks
    assert result.unchanged_chunks == 0
    assert embedder_v2.call_count == result.total_chunks
    assert result.new_version == 2


# 10. partial embedding failure -> failed chunks retryable without redoing successes
def test_partial_failure_is_retried_without_reembedding_successes():
    manifest_store = InMemoryManifestStore()
    vector_store = InMemoryVectorStore()
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    doc = _make_document("doc-i", page_count=5)

    good_embedder = CountingEmbedder()
    IncrementalIndexer(chunker, good_embedder, vector_store, manifest_store).index(doc)

    changed = _make_document("doc-i", page_count=5, changed_pages={
        2: "Page 2 Heading\nRewritten page two.",
        4: "Page 4 Heading\nRewritten page four.",
    })
    # Batch embedding call for the two changed chunks fails outright this run.
    flaky = FlakyEmbedder(fail_on_call=1)
    first_attempt = IncrementalIndexer(chunker, flaky, vector_store, manifest_store).index(changed)

    assert first_attempt.changed_chunks == 2
    assert first_attempt.had_failures
    manifest = manifest_store.get("doc-i")
    failed_ids = [cid for cid, entry in manifest.chunks.items() if entry.status != ChunkStatus.SUCCESS]
    assert len(failed_ids) == 2

    # Retry: same content, a working embedder this time. Only the
    # still-PENDING chunks should be re-embedded - not the 3 unaffected
    # pages' chunks, which were never touched.
    working = CountingEmbedder()
    second_attempt = IncrementalIndexer(chunker, working, vector_store, manifest_store).index(changed)

    assert second_attempt.changed_chunks == 2
    assert not second_attempt.had_failures
    assert working.call_count == 2
    manifest_after = manifest_store.get("doc-i")
    assert all(entry.status == ChunkStatus.SUCCESS for entry in manifest_after.chunks.values())


# 11. multiple documents, only one changed -> only that document's chunks processed
def test_only_the_changed_document_is_reprocessed_among_several():
    manifest_store = InMemoryManifestStore()
    vector_store = InMemoryVectorStore()
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    embedder = CountingEmbedder()
    indexer = IncrementalIndexer(chunker, embedder, vector_store, manifest_store)

    docs = {name: _make_document(name, page_count=5) for name in ("doc-x", "doc-y", "doc-z")}
    for doc in docs.values():
        indexer.index(doc)
    embedder.embedded_texts.clear()

    updated_y = _make_document("doc-y", page_count=5, changed_pages={
        3: "Page 3 Heading\nOnly doc-y's page three actually changed."
    })
    result = indexer.index(updated_y)

    assert result.document_id == "doc-y"
    assert result.changed_chunks == 1
    assert embedder.call_count == 1  # doc-x and doc-z were never touched


# Phase 19: numeric optimization demonstration - many documents/chunks,
# a handful of pages change, assert the exact embedding call count.
def test_performance_optimization_exact_embedding_call_count():
    manifest_store = InMemoryManifestStore()
    vector_store = InMemoryVectorStore()
    chunker = RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    embedder = CountingEmbedder()
    indexer = IncrementalIndexer(chunker, embedder, vector_store, manifest_store)

    # 10 documents x 100 pages x 1 chunk/page (short pages) = 1000 chunks total.
    docs = [_make_document(f"doc-{i}", page_count=100) for i in range(10)]
    total_chunks = 0
    for doc in docs:
        result = indexer.index(doc)
        total_chunks += result.total_chunks

    assert embedder.call_count == total_chunks  # full cost of the old approach

    embedder.embedded_texts.clear()

    # Only 2 pages, in one document, actually change.
    target = docs[3]
    updated = _make_document(target.document_id, page_count=100, changed_pages={
        37: "Page 37 Heading\nRewritten page thirty-seven for the perf test.",
        38: "Page 38 Heading\nRewritten page thirty-eight for the perf test.",
    })
    result = indexer.index(updated)

    assert result.changed_chunks == 2
    assert embedder.call_count == 2  # new approach: 2 calls, not total_chunks
    assert embedder.call_count < total_chunks
