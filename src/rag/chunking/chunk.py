from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_version: int = 1
    # How many times THIS specific chunk's own content has actually
    # changed across ingests, tracked by content-hash lineage rather
    # than position - stays at 1 while the chunk is untouched or merely
    # moved to a different position/index by an edit elsewhere on the
    # page (its content, and therefore its embedding, didn't change),
    # and only increments when this chunk's own text is genuinely
    # different from what last held its identity. See
    # IncrementalIndexer for how lineage is tracked. Always 1 for the
    # non-incremental indexing path (bare RAGService(), no
    # manifest_store) - that path has no history to compare against.
    chunk_version: int = 1
    # Human-readable audit string ("{file_name}:v{document_version}.
    # {chunk_version}:{chunk_id}:{indexed_at date}") - NOT used for
    # identity/lookups (chunk_id alone is, and must stay stable across
    # re-ingests for incremental diffing to work at all - baking a
    # date or version into chunk_id itself would make it change on
    # every ingest regardless of content, defeating that). This field
    # exists purely so a person or log line can see at a glance which
    # file/version/chunk-position/date a given chunk came from without
    # cross-referencing the manifest.
    chunk_label: str | None = None
    chunk_index: int
    page_number: int | None = None
    text: str
    source: str
    document_type: str
    owner: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    indexed_at: datetime | None = None
    parent_section: str | None = None
    content_hash: str | None = None
    chunking_version: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None
    # Document-level authorization fields. All optional and unenforced by
    # default (an empty access_groups list means "everyone can see this
    # chunk") - no tenant/classification business rules are invented here,
    # since none are defined anywhere in this project; a real deployment
    # would populate these at ingest time according to its own actual
    # access model. See RAGService._filter_by_access() for the one rule
    # that IS enforced: a chunk with a non-empty access_groups list is
    # only returned to a caller whose own groups intersect it.
    tenant_id: str | None = None
    access_groups: list[str] = Field(default_factory=list)
    classification: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
