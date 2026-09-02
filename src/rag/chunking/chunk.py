from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_version: int = 1
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
