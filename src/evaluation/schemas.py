from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    relevant_chunk_ids: list[str]
    category: str | None = None
    difficulty: str | None = None


@dataclass(frozen=True)
class GoldenDataset:
    name: str
    queries: list[GoldenQuery]
    description: str | None = None
    source_documents: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    metrics: dict[str, float]
    retrieval_latency_seconds: float
    category: str | None = None
    difficulty: str | None = None


@dataclass(frozen=True)
class LatencyStats:
    average_seconds: float
    median_seconds: float
    p95_seconds: float


@dataclass(frozen=True)
class ExperimentMetadata:
    """
    Per-report provenance - what config produced these numbers, so two
    reports are actually comparable. Not a full experiment-tracking
    platform (see ExperimentTracker below) - just enough to answer
    "what changed between this report and that one."
    """
    timestamp: str
    dataset_name: str
    embedding_provider: str
    embedding_model_name: str | None
    chunk_size: int
    chunk_overlap: int
    retriever: str
    reranker: str | None
    generation_provider: str | None
    guardrails_enabled: bool
    git_commit_hash: str | None


@dataclass(frozen=True)
class EvaluationReport:
    metadata: ExperimentMetadata
    k_values: list[int]
    aggregate_metrics: dict[str, float]
    query_evaluations: list[QueryEvaluation]
    retrieval_latency: LatencyStats


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    current: float
    baseline: float
    delta: float
    delta_percent: float
    is_regression: bool


@dataclass(frozen=True)
class RegressionComparison:
    threshold: float
    deltas: list[MetricDelta]

    @property
    def has_regressions(self) -> bool:
        return any(delta.is_regression for delta in self.deltas)


# ---------------------------------------------------------------------------
# Extension hooks for Layers 2-4. None of these are implemented - they exist
# so those layers can be added later without changing the Layer 1 runner,
# report format, or CLI. A concrete implementation registers by satisfying
# the Protocol; nothing else needs to change.
# ---------------------------------------------------------------------------


class GenerationMetric(Protocol):
    """
    Layer 2 (not implemented): scores a single (query, answer, context)
    triple for generation quality. Concrete implementations - Groundedness,
    Faithfulness, Answer Relevance, Context Precision, Context Recall,
    RAGAS, LLM-as-a-Judge - would each implement this.
    """
    name: str

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunk_texts: list[str]
    ) -> float:
        ...


class SystemMetricsCollector(Protocol):
    """
    Layer 3 (not implemented): collects system-level metrics for a run -
    cost, throughput, memory, token usage, guardrail-specific rates -
    beyond the retrieval latency Layer 1 already measures directly.
    """
    name: str

    def collect(
        self,
        report: EvaluationReport
    ) -> dict[str, float]:
        ...


class ExperimentTracker(Protocol):
    """
    Layer 4 (not implemented): persists and compares EvaluationReports
    across many runs for trend visualization, dashboards, or CI/CD
    regression gating - beyond the single-baseline diff report.py already
    supports (see evaluation.report.compare_reports).
    """

    def record(
        self,
        report: EvaluationReport
    ) -> None:
        ...

    def compare_many(
        self,
        reports: list[EvaluationReport]
    ) -> Any:
        ...
