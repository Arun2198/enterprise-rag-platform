from enum import Enum

from pydantic import BaseModel
from pydantic import Field


class ChunkStatus(str, Enum):
    """
    Per-chunk indexing status within a manifest. PENDING/FAILED chunks are
    exactly what a retry pass re-embeds - SUCCESS chunks are left alone so
    a partial failure never forces redoing work that already landed.
    """
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PENDING = "PENDING"


class ChunkManifestEntry(BaseModel):
    chunk_id: str
    content_hash: str
    status: ChunkStatus = ChunkStatus.PENDING
    error: str | None = None


class PageManifestEntry(BaseModel):
    page_number: int
    page_hash: str
    chunk_ids: list[str] = Field(default_factory=list)


class DocumentManifest(BaseModel):
    """
    What IncrementalIndexer needs to remember about a previously-ingested
    document to answer "what changed" on the next ingest: per-page content
    hashes (to skip untouched pages entirely), per-chunk content hashes
    and status (to skip untouched chunks within a changed page, and to
    retry only chunks that failed to embed/index last time), and the
    embedding fingerprint the chunks were actually embedded with (so a
    model/provider/dimension change is detected even when nothing in the
    document's own content changed).
    """
    document_id: str
    document_version: int = 1
    document_hash: str
    embedding_fingerprint: str
    chunking_version: str
    pages: list[PageManifestEntry] = Field(default_factory=list)
    chunks: dict[str, ChunkManifestEntry] = Field(default_factory=dict)
    updated_at: str | None = None
