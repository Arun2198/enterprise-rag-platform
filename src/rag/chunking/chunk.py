from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    source: str
    document_type: str
    owner: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    parent_section: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
