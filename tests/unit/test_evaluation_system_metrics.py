from datetime import datetime
from datetime import timezone

from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import LatencyStats
from evaluation.schemas import QueryEvaluation
from evaluation.system_metrics import DefaultSystemMetricsCollector


def _metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name="test",
        embedding_provider="hashing",
        embedding_model_name=None,
        chunk_size=900,
        chunk_overlap=120,
        retriever="hybrid",
        reranker=None,
        generation_provider=None,
        guardrails_enabled=False,
        git_commit_hash=None
    )


def _report(query_evaluations: list[QueryEvaluation]) -> EvaluationReport:
    return EvaluationReport(
        metadata=_metadata(),
        k_values=[1],
        aggregate_metrics={},
        query_evaluations=query_evaluations,
        retrieval_latency=LatencyStats(average_seconds=0.0, median_seconds=0.0, p95_seconds=0.0)
    )


def _query_evaluation(
    retrieval_latency_seconds: float = 0.1,
    answer: str | None = None
) -> QueryEvaluation:
    return QueryEvaluation(
        query_id="Q1",
        query="what is the leave policy",
        retrieved_chunk_ids=["a"],
        relevant_chunk_ids=["a"],
        metrics={},
        retrieval_latency_seconds=retrieval_latency_seconds,
        answer=answer
    )


def test_collect_reports_query_count_and_throughput():

    report = _report([_query_evaluation(retrieval_latency_seconds=0.5)] * 2)
    collector = DefaultSystemMetricsCollector()

    metrics = collector.collect(report)

    assert metrics["queries_evaluated"] == 2.0
    assert metrics["retrieval_throughput_qps"] == 2.0 / 1.0


def test_collect_handles_zero_latency_without_dividing_by_zero():

    report = _report([_query_evaluation(retrieval_latency_seconds=0.0)])
    collector = DefaultSystemMetricsCollector()

    metrics = collector.collect(report)

    assert metrics["retrieval_throughput_qps"] == 0.0


def test_collect_estimates_tokens_and_cost_from_answers():

    report = _report([_query_evaluation(answer="a" * 40)])
    collector = DefaultSystemMetricsCollector(cost_per_1k_completion_tokens=10.0)

    metrics = collector.collect(report)

    assert metrics["estimated_completion_tokens_total"] == 10.0  # 40 chars / 4
    assert metrics["estimated_completion_cost_usd"] == 0.1  # 10 tokens / 1000 * 10.0


def test_collect_omits_token_metrics_when_no_answers_present():

    report = _report([_query_evaluation(answer=None)])
    collector = DefaultSystemMetricsCollector()

    metrics = collector.collect(report)

    assert "estimated_completion_tokens_total" not in metrics


def test_collect_includes_run_duration_and_memory_only_when_supplied():

    report = _report([_query_evaluation()])
    collector = DefaultSystemMetricsCollector()

    without_extras = collector.collect(report)
    assert "run_duration_seconds" not in without_extras
    assert "peak_memory_mb" not in without_extras

    with_extras = collector.collect(report, run_duration_seconds=2.0, peak_memory_mb=15.5)
    assert with_extras["run_duration_seconds"] == 2.0
    assert with_extras["peak_memory_mb"] == 15.5
    assert with_extras["overall_throughput_qps"] == 0.5


def test_collect_does_not_report_a_guardrail_trigger_rate():

    report = _report([_query_evaluation()])
    collector = DefaultSystemMetricsCollector()

    metrics = collector.collect(report)

    assert not any("guardrail" in key for key in metrics)
