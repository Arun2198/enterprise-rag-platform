from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

MAX_QUERY_LENGTH = 2000
MAX_FILE_PATHS_PER_REQUEST = 50


class IngestRequest(BaseModel):
    file_paths: list[str] = Field(min_length=1, max_length=MAX_FILE_PATHS_PER_REQUEST)
    document_ids: list[str] = Field(
        min_length=1,
        description=(
            "Required stable identity per file_paths entry - same length, "
            "index-aligned. Mandatory (not derived from the filename) so "
            "a renamed file is never mistaken for a brand-new document: "
            "pass the same document_id on every ingest of the same "
            "logical document, including after a rename."
        )
    )

    @model_validator(mode="after")
    def _document_ids_length_must_match_file_paths(self) -> "IngestRequest":
        if len(self.document_ids) != len(self.file_paths):
            raise ValueError(
                "document_ids must have the same length as file_paths "
                f"({len(self.document_ids)} != {len(self.file_paths)})"
            )
        return self


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_chunks: int
    errors: list[str] = Field(default_factory=list)


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted_chunks: int


class ReindexRequest(BaseModel):
    file_path: str = Field(min_length=1)
    document_id: str = Field(
        min_length=1,
        description="Required stable identity - same meaning as IngestRequest.document_ids."
    )


class DocumentUploadResponse(BaseModel):
    document_id: str
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    error: str | None = None
    created_at: str
    updated_at: str


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    top_k: int = Field(default=5, ge=1, le=20)
    client_id: str | None = Field(
        default=None,
        description="Stable per-caller id used to bucket feature-flag canary rollouts consistently"
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Existing conversation to continue - when set, prior turns from this "
            "conversation are threaded into the prompt as history and the new "
            "query/answer pair is appended to it. Omit to ask a one-off question "
            "with no history."
        )
    )


class Source(BaseModel):
    document_id: str
    document_version: int = 1
    chunk_id: str
    section: str | None = None
    source: str
    score: float
    retrieval_method: str = "dense"
    rank: int = 0
    text: str


class CitationResponse(BaseModel):
    source_number: int
    valid: bool
    document_id: str | None = None
    document_version: int | None = None
    chunk_id: str | None = None
    section: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    # How much retrieved evidence actually supports the answer text
    # (from HallucinationDetector), NOT the search engine's own relevance
    # score for the top chunk - those measure different things (did we
    # find the right document vs. does the generated answer actually say
    # what the document says). None when the hallucination guard is
    # disabled, since there's no real groundedness signal to report then.
    groundedness: float | None = None
    confidence: float
    # [Source N] markers parsed out of the answer text and resolved
    # against `sources` - empty for ExtractiveAnswerer (no inline
    # citations to parse) and for LLM answers that made no claims citing
    # a specific source. See rag.generation.citations.extract_citations.
    citations: list[CitationResponse] = Field(default_factory=list)
    guardrail_flags: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None


class CandidateTraceResponse(BaseModel):
    chunk_id: str
    score: float
    rank: int


class RetrievalTraceResponse(BaseModel):
    query: str
    embedding_provider: str | None = None
    embedding_dimensions: int | None = None
    dense_candidates: list[CandidateTraceResponse] = Field(default_factory=list)
    bm25_candidates: list[CandidateTraceResponse] = Field(default_factory=list)
    fused_candidates: list[CandidateTraceResponse] = Field(default_factory=list)
    reranker_used: bool = False
    reranked_candidates: list[CandidateTraceResponse] = Field(default_factory=list)
    final_chunk_ids: list[str] = Field(default_factory=list)
    generation_provider: str | None = None
    groundedness: float | None = None
    guardrail_findings: list[dict[str, Any]] = Field(default_factory=list)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)


class AskDebugResponse(BaseModel):
    response: AskResponse
    trace: RetrievalTraceResponse


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


class BackupRestoreRequest(BaseModel):
    snapshot_id: str = Field(min_length=1)


class BackupRestoreResponse(BaseModel):
    snapshot_id: str
    components: list[str]


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ConversationSummaryResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int


class ConversationTurnResponse(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    turns: list[ConversationTurnResponse]
