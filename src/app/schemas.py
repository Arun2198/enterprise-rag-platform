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
    client_id: str | None = Field(
        default=None,
        description="Stable per-caller id used to bucket feature-flag canary rollouts consistently"
    )


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


class FeatureFlagResponse(BaseModel):
    name: str
    enabled: bool
    rollout_percentage: float
    shadow: bool
    description: str | None
    updated_at: str | None


class FeatureFlagUpdateRequest(BaseModel):
    enabled: bool | None = None
    rollout_percentage: float | None = Field(default=None, ge=0.0, le=100.0)


class ScheduledJobResponse(BaseModel):
    job_id: str
    name: str
    enabled: bool
    interval_seconds: float
    next_run_at: float


class JobRunResponse(BaseModel):
    job_id: str
    started_at: str
    finished_at: str | None
    success: bool | None
    error: str | None
