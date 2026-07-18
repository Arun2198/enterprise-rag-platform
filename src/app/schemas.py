from typing import Any

from pydantic import BaseModel
from pydantic import Field


class IngestRequest(BaseModel):
    file_paths: list[str] = Field(min_length=1)


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    errors: list[str] = Field(default_factory=list)


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    document_id: str
    chunk_id: str
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    confidence: float
    guardrail_flags: dict[str, Any] = Field(default_factory=dict)
