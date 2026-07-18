import csv
import io
import json
import logging
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from pathlib import Path

from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import LatencyStats
from evaluation.schemas import MetricDelta
from evaluation.schemas import QueryEvaluation
from evaluation.schemas import RegressionComparison

logger = logging.getLogger(__name__)

DEFAULT_REGRESSION_THRESHOLD = 0.02


def render_console(
    report: EvaluationReport
) -> str:
    lines = ["=" * 49]
    lines.append(f"Dataset: {report.metadata.dataset_name}")
    lines.append(f"Queries: {len(report.query_evaluations)}")
    lines.append(f"Timestamp: {report.metadata.timestamp}")
    lines.append(f"Git commit: {report.metadata.git_commit_hash or 'unknown'}")
    lines.append("-" * 49)

    for k in report.k_values:
        lines.append(f"Recall@{k}: {report.aggregate_metrics[f'recall@{k}']:.4f}")
        lines.append(f"Precision@{k}: {report.aggregate_metrics[f'precision@{k}']:.4f}")
        lines.append(f"Hit Rate@{k}: {report.aggregate_metrics[f'hit_rate@{k}']:.4f}")
        lines.append(f"NDCG@{k}: {report.aggregate_metrics[f'ndcg@{k}']:.4f}")

    lines.append("-" * 49)
    lines.append(f"MRR: {report.aggregate_metrics['mrr']:.4f}")
    lines.append(f"Average Rank: {report.aggregate_metrics['average_rank']:.2f}")
    lines.append(
        f"Average Retrieved Documents: "
        f"{report.aggregate_metrics['average_retrieved_documents']:.2f}"
    )
    lines.append("-" * 49)
    lines.append(
        f"Average Retrieval Time: {report.retrieval_latency.average_seconds * 1000:.1f} ms"
    )
    lines.append(
        f"Median Retrieval Time: {report.retrieval_latency.median_seconds * 1000:.1f} ms"
    )
    lines.append(
        f"P95 Retrieval Time: {report.retrieval_latency.p95_seconds * 1000:.1f} ms"
    )
    lines.append("=" * 49)

    return "\n".join(lines)


def to_json_dict(
    report: EvaluationReport
) -> dict:
    return asdict(report)


def write_json_report(
    report: EvaluationReport,
    output_dir: str
) -> Path:
    path = _report_path(report, output_dir, extension="json")
    path.write_text(
        json.dumps(to_json_dict(report), indent=2),
        encoding="utf-8"
    )
    logger.info("report_written", extra={"format": "json", "path": str(path)})
    return path


def write_csv_report(
    report: EvaluationReport,
    output_dir: str
) -> Path:
    path = _report_path(report, output_dir, extension="csv")
    path.write_text(render_csv(report), encoding="utf-8")
    logger.info("report_written", extra={"format": "csv", "path": str(path)})
    return path


def render_csv(
    report: EvaluationReport
) -> str:
    buffer = io.StringIO()
    metric_columns = sorted(report.query_evaluations[0].metrics) if report.query_evaluations else []
    writer = csv.writer(buffer)
    writer.writerow(
        ["query_id", "query", "category", "difficulty", "retrieval_latency_seconds"]
        + metric_columns
    )

    for query_evaluation in report.query_evaluations:
        writer.writerow(
            [
                query_evaluation.query_id,
                query_evaluation.query,
                query_evaluation.category or "",
                query_evaluation.difficulty or "",
                f"{query_evaluation.retrieval_latency_seconds:.6f}"
            ]
            + [f"{query_evaluation.metrics[column]:.4f}" for column in metric_columns]
        )

    return buffer.getvalue()


def load_json_report(
    path: str
) -> EvaluationReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    return EvaluationReport(
        metadata=ExperimentMetadata(**raw["metadata"]),
        k_values=raw["k_values"],
        aggregate_metrics=raw["aggregate_metrics"],
        query_evaluations=[
            QueryEvaluation(**query_evaluation)
            for query_evaluation in raw["query_evaluations"]
        ],
        retrieval_latency=LatencyStats(**raw["retrieval_latency"])
    )


def compare_reports(
    current: EvaluationReport,
    baseline: EvaluationReport,
    threshold: float = DEFAULT_REGRESSION_THRESHOLD
) -> RegressionComparison:
    """
    Diffs two already-produced reports metric by metric. This is
    intentionally just a two-report diff, not a trend/dashboard/CI-CD
    system - see evaluation.schemas.ExperimentTracker for that extension
    point.
    """
    deltas = []

    for metric_name in sorted(current.aggregate_metrics):
        current_value = current.aggregate_metrics[metric_name]
        baseline_value = baseline.aggregate_metrics.get(metric_name)

        if baseline_value is None:
            continue

        delta = current_value - baseline_value
        delta_percent = (delta / baseline_value * 100) if baseline_value else 0.0

        deltas.append(
            MetricDelta(
                metric=metric_name,
                current=current_value,
                baseline=baseline_value,
                delta=delta,
                delta_percent=delta_percent,
                is_regression=delta < -threshold
            )
        )

    return RegressionComparison(threshold=threshold, deltas=deltas)


def render_comparison_console(
    comparison: RegressionComparison
) -> str:
    lines = ["=" * 65]
    lines.append(f"{'Metric':<28}{'Current':>10}{'Baseline':>10}{'Change':>10}{'':>7}")
    lines.append("-" * 65)

    for metric_delta in comparison.deltas:
        flag = " REGRESSION" if metric_delta.is_regression else ""
        lines.append(
            f"{metric_delta.metric:<28}"
            f"{metric_delta.current:>10.4f}"
            f"{metric_delta.baseline:>10.4f}"
            f"{metric_delta.delta_percent:>+9.1f}%{flag}"
        )

    lines.append("=" * 65)

    if comparison.has_regressions:
        lines.append(
            f"REGRESSIONS DETECTED (threshold={comparison.threshold:.2%})"
        )
    else:
        lines.append(f"No regressions (threshold={comparison.threshold:.2%})")

    return "\n".join(lines)


def _report_path(
    report: EvaluationReport,
    output_dir: str,
    extension: str
) -> Path:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = report.metadata.dataset_name.replace(" ", "_")
    return directory / f"{safe_name}_{timestamp}.{extension}"
