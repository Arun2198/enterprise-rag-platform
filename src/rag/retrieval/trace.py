from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class CandidateTrace:
    chunk_id: str
    score: float
    rank: int


@dataclass
class RetrievalTrace:
    """
    Per-request debugging trace covering every retrieval/generation stage -
    what went into and came out of dense search, BM25, RRF fusion,
    reranking, and generation, plus per-stage latency. Built up by
    RAGService.ask_with_trace() (never by the normal ask() path, so there's
    no overhead for regular callers) and gated behind a permission check at
    the API layer since it exposes internal scoring detail not meant for
    every caller.
    """
    query: str
    embedding_provider: str | None = None
    embedding_dimensions: int | None = None
    dense_candidates: list[CandidateTrace] = field(default_factory=list)
    bm25_candidates: list[CandidateTrace] = field(default_factory=list)
    fused_candidates: list[CandidateTrace] = field(default_factory=list)
    reranker_used: bool = False
    reranked_candidates: list[CandidateTrace] = field(default_factory=list)
    final_chunk_ids: list[str] = field(default_factory=list)
    generation_provider: str | None = None
    groundedness: float | None = None
    guardrail_findings: list[dict] = field(default_factory=list)
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
