import hashlib
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from ingestion.contracts.document import Document
from ingestion.contracts.manifest import ChunkManifestEntry
from ingestion.contracts.manifest import ChunkStatus
from ingestion.contracts.manifest import DocumentManifest
from ingestion.contracts.manifest import PageManifestEntry
from ingestion.manifest_store import ManifestStore
from rag.chunking.chunk import Chunk
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.base import Embedder
from rag.vector_store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IncrementalIndexResult:
    """
    Everything Phase 16's structured-logging requirement asks for, plus
    what the API response needs. `indexed_chunk_count` is the document's
    total live chunk count after this run (matches the old index_document
    return value's meaning) - changed_chunks/unchanged_chunks/
    reused_chunks/deleted_chunks are the finer-grained breakdown of how
    that total was reached. `changed_chunks` is specifically the count
    that triggered a real embedding-model call; `reused_chunks` is
    content that moved position on the page (an edit elsewhere shifted
    it) but whose text - and therefore embedding - didn't actually
    change, so its existing embedding was copied forward instead of
    recomputed.
    """
    document_id: str
    previous_version: int | None
    new_version: int
    total_pages: int
    changed_pages: int
    unchanged_pages: int
    deleted_pages: int
    total_chunks: int
    changed_chunks: int
    unchanged_chunks: int
    reused_chunks: int
    deleted_chunks: int
    embedding_calls: int
    embedding_skipped_count: int
    embedding_failures: list[str] = field(default_factory=list)
    processing_duration_seconds: float = 0.0
    indexed_chunk_count: int = 0

    @property
    def had_failures(self) -> bool:
        return len(self.embedding_failures) > 0


def _normalize_for_hash(text: str) -> str:
    # Whitespace-insensitive so re-extracting the same PDF page (which can
    # vary in incidental whitespace between pypdf runs/versions) doesn't
    # register as a content change.
    return " ".join(text.split())


def compute_page_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def compute_document_hash(pages: list[str]) -> str:
    return hashlib.sha256(
        "\n".join(_normalize_for_hash(page) for page in pages).encode("utf-8")
    ).hexdigest()


def embedding_fingerprint_for(embedder: Embedder) -> str:
    model_name = getattr(embedder, "model_name", embedder.provider_name)
    return f"{embedder.provider_name}:{model_name}:{embedder.dimensions}"


class IncrementalIndexer:
    """
    Diffs a freshly-parsed Document against its stored DocumentManifest
    (if any) and touches only the pages/chunks that actually changed:
    unchanged chunks are never re-embedded and never re-written to the
    vector store, changed/new chunks are embedded and upserted, chunks
    whose content merely moved to a different position on the page (an
    edit elsewhere shifted the sentence-packing boundaries) get their
    existing embedding copied forward instead of recomputed, and chunks
    that no longer exist (a page or the whole tail of a document was
    removed) are deleted. See CLAUDE.md's incremental-re-embedding
    section for the full design writeup.

    Change detection is entirely chunk-hash-driven, not page-hash-driven -
    page hashes are computed and reported for the Phase 16 log summary,
    but the actual embed/skip decision is a content_hash comparison per
    chunk. This naturally satisfies the finer-grained requirement of "a
    chunk within a changed page can still be unchanged" without needing
    separate page-level and chunk-level code paths - it also means a page
    whose extracted text shifted by pure whitespace (a re-extraction
    quirk, not a real edit) still skips re-embedding every one of its
    chunks, since content_hash normalizes whitespace before hashing.

    Content-addressed reuse (not just position-addressed): chunking is
    greedy sentence-packing up to chunk_size, so a length-changing edit
    to one sentence can shift every later chunk's boundaries on that
    page - the sentences after the edit are unchanged, but they now fall
    into a differently-positioned chunk with a different chunk_id. A
    pure "same chunk_id, same hash" comparison would treat all of those
    as new and re-embed the whole rest of the page. Instead, any new
    chunk whose exact text (content_hash) matches some *not yet reused*
    chunk from the previous manifest for the same page - regardless of
    position - has its embedding fetched back out of the vector store
    (VectorStore.get_embedding) and copied to the new chunk_id rather
    than recomputed. Only chunks whose text is genuinely different from
    anything in the previous version of that page ever trigger a real
    embedding call.
    """

    def __init__(
        self,
        chunker: RecursiveChunker,
        embedder: Embedder,
        vector_store: VectorStore,
        manifest_store: ManifestStore
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.manifest_store = manifest_store

    def index(
        self,
        document: Document
    ) -> IncrementalIndexResult | None:
        """
        Returns None if chunking failed (mirrors RAGService.index_document's
        existing None-on-chunk-failure contract), otherwise the full
        result. Never raises on a partial embedding/indexing failure -
        failed chunks are recorded in the manifest as FAILED so the next
        ingest of the same content retries only them.
        """
        started = time.monotonic()

        chunk_result = self.chunker.chunk(document)

        if not chunk_result.success or chunk_result.data is None:
            return None

        new_chunks = chunk_result.data
        pages = document.pages if document.pages is not None else [document.content]
        fingerprint = embedding_fingerprint_for(self.embedder)
        chunking_version = self.chunker.chunking_version
        document_hash = compute_document_hash(pages)
        file_name = document.metadata.get("file_name", document.source)

        previous = self.manifest_store.get(document.document_id)
        fingerprint_changed = (
            previous is not None
            and (
                previous.embedding_fingerprint != fingerprint
                or previous.chunking_version != chunking_version
            )
        )
        previous_chunks = {} if previous is None or fingerprint_changed else previous.chunks
        previous_pages = {} if previous is None else {p.page_number: p for p in previous.pages}

        new_chunk_ids = {c.chunk_id for c in new_chunks}

        # Reverse index for content-addressed reuse: hash -> old entries
        # that still hold that exact content, grouped by page (a chunk
        # moving to a different *page* isn't something an edit to one
        # page's text can cause, so this stays page-scoped rather than
        # matching across the whole document).
        hash_pool_by_page: dict[int, dict[str, list[ChunkManifestEntry]]] = {}
        for chunk_id, entry in previous_chunks.items():
            page_number = self._page_number_from_chunk_id(chunk_id)
            if entry.status != ChunkStatus.SUCCESS:
                continue
            hash_pool_by_page.setdefault(page_number, {}).setdefault(entry.content_hash, []).append(entry)

        claimed_old_ids: set[str] = set()
        to_embed: list[Chunk] = []
        to_reuse: list[tuple[Chunk, str]] = []  # (new_chunk, source_old_chunk_id)
        chunk_version_by_id: dict[str, int] = {}
        unchanged_count = 0

        for chunk in new_chunks:
            old_entry = previous_chunks.get(chunk.chunk_id)

            if (
                old_entry is not None
                and old_entry.content_hash == chunk.content_hash
                and old_entry.status == ChunkStatus.SUCCESS
            ):
                unchanged_count += 1
                chunk_version_by_id[chunk.chunk_id] = old_entry.chunk_version
                claimed_old_ids.add(chunk.chunk_id)
                continue

            page_pool = hash_pool_by_page.get(chunk.page_number or 1, {})
            candidates = page_pool.get(chunk.content_hash or "", [])
            reusable = next((c for c in candidates if c.chunk_id not in claimed_old_ids), None)

            if reusable is not None:
                to_reuse.append((chunk, reusable.chunk_id))
                chunk_version_by_id[chunk.chunk_id] = reusable.chunk_version
                claimed_old_ids.add(reusable.chunk_id)
                continue

            to_embed.append(chunk)
            chunk_version_by_id[chunk.chunk_id] = (
                (old_entry.chunk_version + 1) if old_entry is not None else 1
            )

        stale_chunk_ids = set(previous_chunks.keys()) - new_chunk_ids

        # Page-level stats are informational (Phase 16's log shape asks
        # for them explicitly) - computed independently of the embed
        # decision above, which is chunk-hash-driven.
        new_page_numbers = {i + 1 for i, text in enumerate(pages) if text.strip()}
        changed_page_count = 0
        unchanged_page_count = 0
        for page_number in new_page_numbers:
            page_text = pages[page_number - 1]
            old_page = previous_pages.get(page_number)
            page_hash = compute_page_hash(page_text)
            if old_page is not None and old_page.page_hash == page_hash and not fingerprint_changed:
                unchanged_page_count += 1
            else:
                changed_page_count += 1
        deleted_page_count = len(set(previous_pages.keys()) - new_page_numbers)

        embedding_calls = 0
        failures: list[str] = []
        status_by_chunk_id: dict[str, ChunkStatus] = {}
        error_by_chunk_id: dict[str, str] = {}
        next_version = self._next_version(previous, document_hash, fingerprint_changed)
        indexed_at = datetime.now(timezone.utc)

        # Fetch every reused embedding BEFORE writing anything - a
        # source chunk's old id can coincide with a *different* new
        # chunk's id (a one-position shift), so writing as we go could
        # overwrite a still-unread source entry before its own reuse
        # lookup runs. All reads happen first, all writes happen after,
        # so ordering never corrupts a read that hasn't happened yet.
        reused_embeddings: dict[str, list[float]] = {}
        reuse_fallback: list[Chunk] = []
        for chunk, source_old_id in to_reuse:
            embedding = None
            try:
                embedding = self.vector_store.get_embedding(source_old_id)
            except Exception as ex:
                logger.warning(
                    "incremental_ingest_embedding_reuse_fetch_failed",
                    extra={"document_id": document.document_id, "source_chunk_id": source_old_id, "error": str(ex)}
                )

            if embedding is None:
                reuse_fallback.append(chunk)
            else:
                reused_embeddings[chunk.chunk_id] = embedding

        reused_count = len(to_reuse) - len(reuse_fallback)
        to_embed.extend(reuse_fallback)

        for chunk_id, embedding in reused_embeddings.items():
            chunk = next(c for c in new_chunks if c.chunk_id == chunk_id)
            stored_chunk = self._stamp_chunk(
                chunk, next_version, chunk_version_by_id[chunk.chunk_id], indexed_at, file_name
            )
            try:
                self.vector_store.add(stored_chunk, embedding)
                status_by_chunk_id[chunk.chunk_id] = ChunkStatus.SUCCESS
            except Exception as ex:
                logger.warning(
                    "incremental_ingest_chunk_index_failed",
                    extra={"document_id": document.document_id, "chunk_id": chunk.chunk_id, "error": str(ex)}
                )
                status_by_chunk_id[chunk.chunk_id] = ChunkStatus.FAILED
                error_by_chunk_id[chunk.chunk_id] = str(ex)
                failures.append(f"{chunk.chunk_id}: INDEX_FAILED {ex}")

        if to_embed:
            try:
                embeddings = self.embedder.embed_batch([c.text for c in to_embed])
                embedding_calls = len(to_embed)
            except Exception as ex:
                # The whole batch call failed before any individual chunk
                # could be attempted - every chunk in this run is PENDING,
                # not FAILED, since we never actually tried each one
                # individually; a retry will simply re-attempt the batch.
                logger.warning(
                    "incremental_ingest_embedding_batch_failed",
                    extra={"document_id": document.document_id, "error": str(ex)}
                )
                for chunk in to_embed:
                    status_by_chunk_id[chunk.chunk_id] = ChunkStatus.PENDING
                    failures.append(f"{chunk.chunk_id}: EMBEDDING_FAILED {ex}")
                embeddings = None

            if embeddings is not None:
                for chunk, embedding in zip(to_embed, embeddings, strict=True):
                    stored_chunk = self._stamp_chunk(
                        chunk, next_version, chunk_version_by_id[chunk.chunk_id], indexed_at, file_name
                    )
                    try:
                        self.vector_store.add(stored_chunk, embedding)
                        status_by_chunk_id[chunk.chunk_id] = ChunkStatus.SUCCESS
                    except Exception as ex:
                        logger.warning(
                            "incremental_ingest_chunk_index_failed",
                            extra={"document_id": document.document_id, "chunk_id": chunk.chunk_id, "error": str(ex)}
                        )
                        status_by_chunk_id[chunk.chunk_id] = ChunkStatus.FAILED
                        error_by_chunk_id[chunk.chunk_id] = str(ex)
                        failures.append(f"{chunk.chunk_id}: INDEX_FAILED {ex}")

        for stale_chunk_id in stale_chunk_ids:
            try:
                self.vector_store.delete(stale_chunk_id)
            except Exception as ex:
                logger.warning(
                    "incremental_ingest_stale_chunk_delete_failed",
                    extra={"document_id": document.document_id, "chunk_id": stale_chunk_id, "error": str(ex)}
                )

        manifest = self._build_manifest(
            document=document,
            new_chunks=new_chunks,
            pages=pages,
            new_page_numbers=new_page_numbers,
            previous_chunks=previous_chunks,
            status_by_chunk_id=status_by_chunk_id,
            error_by_chunk_id=error_by_chunk_id,
            chunk_version_by_id=chunk_version_by_id,
            document_hash=document_hash,
            fingerprint=fingerprint,
            chunking_version=chunking_version,
            version=next_version
        )
        self.manifest_store.put(manifest)

        indexed_chunk_count = sum(
            1 for c in new_chunks
            if manifest.chunks.get(c.chunk_id) and manifest.chunks[c.chunk_id].status == ChunkStatus.SUCCESS
        )

        result = IncrementalIndexResult(
            document_id=document.document_id,
            previous_version=previous.document_version if previous else None,
            new_version=next_version,
            total_pages=len(new_page_numbers),
            changed_pages=changed_page_count,
            unchanged_pages=unchanged_page_count,
            deleted_pages=deleted_page_count,
            total_chunks=len(new_chunks),
            changed_chunks=len(to_embed),
            unchanged_chunks=unchanged_count,
            reused_chunks=reused_count,
            deleted_chunks=len(stale_chunk_ids),
            embedding_calls=embedding_calls,
            embedding_skipped_count=unchanged_count + reused_count,
            embedding_failures=failures,
            processing_duration_seconds=time.monotonic() - started,
            indexed_chunk_count=indexed_chunk_count
        )

        logger.info(
            "incremental_ingest_summary",
            extra={
                "document_id": result.document_id,
                "previous_version": result.previous_version,
                "new_version": result.new_version,
                "total_pages": result.total_pages,
                "changed_pages": result.changed_pages,
                "unchanged_pages": result.unchanged_pages,
                "deleted_pages": result.deleted_pages,
                "total_chunks": result.total_chunks,
                "changed_chunks": result.changed_chunks,
                "unchanged_chunks": result.unchanged_chunks,
                "reused_chunks": result.reused_chunks,
                "deleted_chunks": result.deleted_chunks,
                "embedding_count": result.embedding_calls,
                "embedding_skipped_count": result.embedding_skipped_count,
                "embedding_failures": len(result.embedding_failures),
                "processing_duration_seconds": round(result.processing_duration_seconds, 4)
            }
        )

        return result

    def _stamp_chunk(
        self,
        chunk: Chunk,
        document_version: int,
        chunk_version: int,
        indexed_at: datetime,
        file_name: str
    ) -> Chunk:
        return chunk.model_copy(update={
            "embedding_provider": self.embedder.provider_name,
            "embedding_model": getattr(self.embedder, "model_name", None),
            "embedding_version": str(self.embedder.dimensions),
            "document_version": document_version,
            "chunk_version": chunk_version,
            "indexed_at": indexed_at,
            "chunk_label": (
                f"{file_name}:v{document_version}.{chunk_version}:"
                f"{chunk.chunk_id}:{indexed_at.date().isoformat()}"
            )
        })

    def _page_number_from_chunk_id(
        self,
        chunk_id: str
    ) -> int:
        # chunk_id shape is "{document_id}:p{page}:c{index}" - the
        # page-scoping for reuse only needs the page segment, not a full
        # parse of the id, and falls back to 1 for any id predating this
        # scheme (defensive, not expected in practice).
        try:
            page_segment = chunk_id.rsplit(":", 2)[-2]
            return int(page_segment.removeprefix("p"))
        except (IndexError, ValueError):
            return 1

    def _next_version(
        self,
        previous: DocumentManifest | None,
        document_hash: str,
        fingerprint_changed: bool
    ) -> int:
        if previous is None:
            return 1
        if previous.document_hash == document_hash and not fingerprint_changed:
            return previous.document_version
        return previous.document_version + 1

    def _build_manifest(
        self,
        document: Document,
        new_chunks: list[Chunk],
        pages: list[str],
        new_page_numbers: set[int],
        previous_chunks: dict[str, ChunkManifestEntry],
        status_by_chunk_id: dict[str, ChunkStatus],
        error_by_chunk_id: dict[str, str],
        chunk_version_by_id: dict[str, int],
        document_hash: str,
        fingerprint: str,
        chunking_version: str,
        version: int
    ) -> DocumentManifest:
        chunk_entries: dict[str, ChunkManifestEntry] = {}
        page_chunk_ids: dict[int, list[str]] = {p: [] for p in sorted(new_page_numbers)}

        for chunk in new_chunks:
            page_chunk_ids.setdefault(chunk.page_number or 1, []).append(chunk.chunk_id)

            if chunk.chunk_id in status_by_chunk_id:
                status = status_by_chunk_id[chunk.chunk_id]
            elif chunk.chunk_id in previous_chunks:
                # Unchanged - carry the previous SUCCESS entry forward.
                status = previous_chunks[chunk.chunk_id].status
            else:
                status = ChunkStatus.PENDING

            chunk_entries[chunk.chunk_id] = ChunkManifestEntry(
                chunk_id=chunk.chunk_id,
                content_hash=chunk.content_hash or "",
                chunk_version=chunk_version_by_id.get(chunk.chunk_id, 1),
                status=status,
                error=error_by_chunk_id.get(chunk.chunk_id)
            )

        page_entries = [
            PageManifestEntry(
                page_number=page_number,
                page_hash=compute_page_hash(pages[page_number - 1]),
                chunk_ids=page_chunk_ids.get(page_number, [])
            )
            for page_number in sorted(new_page_numbers)
        ]

        return DocumentManifest(
            document_id=document.document_id,
            document_version=version,
            document_hash=document_hash,
            embedding_fingerprint=fingerprint,
            chunking_version=chunking_version,
            pages=page_entries,
            chunks=chunk_entries,
            updated_at=datetime.now(timezone.utc).isoformat()
        )
