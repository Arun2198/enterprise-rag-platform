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
    answer: str | None = None
    generation_metrics: dict[str, float] = field(default_factory=dict)


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


@dataclass(frozen=True)
class ExperimentRecord:
    """
    One tracked run in Layer 4's history - just enough of an
    EvaluationReport to trend aggregate metrics over time and identify
    what produced them. Not the full report (that's already saved
    separately by --json/--csv per run); keeping this record small is
    what makes an ever-growing history file stay cheap to read.
    """
    timestamp: str
    dataset_name: str
    git_commit_hash: str | None
    embedding_provider: str
    reranker: str | None
    generation_provider: str | None
    aggregate_metrics: dict[str, float]


@dataclass(frozen=True)
class MetricTrendPoint:
    timestamp: str
    value: float
    git_commit_hash: str | None


@dataclass(frozen=True)
class MetricTrend:
    metric: str
    points: list[MetricTrendPoint]

    @property
    def latest(self) -> float | None:
        return self.points[-1].value if self.points else None

    @property
    def delta_from_previous(self) -> float | None:
        if len(self.points) < 2:
            return None

        return self.points[-1].value - self.points[-2].value


# ---------------------------------------------------------------------------
# Extension hooks for Layers 2-4, plus concrete implementations. A new
# implementation just needs to satisfy the Protocol shape and get wired in
# where its layer is used - the runner, report format, and CLI don't change.
# ---------------------------------------------------------------------------


class GenerationMetric(Protocol):
    """
    Layer 2: scores a single (query, answer, context) triple for generation
    quality. Concrete implementations live in evaluation.generation_metrics -
    GroundednessMetric, AnswerRelevanceMetric, ContextRelevanceMetric
    (deterministic, no LLM call) and LLMJudgeGenerationMetric (opt-in,
    reuses the OpenAI-compatible client pattern). RAGAS-style Context
    Precision/Context Recall are intentionally not duplicated here - Layer 1
    already computes those against the golden dataset's relevant_chunk_ids
    (see metrics.recall_at_k/precision_at_k), which is a more reliable
    signal than judging chunk relevance from answer text alone.
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
    Layer 3: collects system-level metrics for a run - cost, throughput,
    memory, token usage, guardrail-specific rates - beyond the retrieval
    latency Layer 1 already measures directly. Concrete implementation:
    evaluation.system_metrics.DefaultSystemMetricsCollector.
    """
    name: str

    def collect(
        self,
        report: EvaluationReport
    ) -> dict[str, float]:
        ...


class ExperimentTracker(Protocol):
    """
    Layer 4: persists and compares EvaluationReports across many runs for
    trend visualization or CI/CD regression gating - beyond the
    single-baseline diff report.py already supports (see
    evaluation.report.compare_reports). Concrete implementation:
    evaluation.experiment_tracker.LocalExperimentTracker (append-only JSON
    history file, no dashboard/UI - see its docstring for what "trend
    visualization" means here).
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
