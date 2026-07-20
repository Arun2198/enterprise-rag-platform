import json
import logging
from dataclasses import asdict
from pathlib import Path

from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentRecord
from evaluation.schemas import MetricTrend
from evaluation.schemas import MetricTrendPoint

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_PATH = "evaluation/reports/experiment_history.json"


class LocalExperimentTracker:
    """
    Layer 4: append-only local run history for trend visualization and
    CI/CD regression gating across many runs - the thing report.py's
    compare_reports() deliberately doesn't do (that's a two-report diff).
    "Trend visualization" here means a console table and the raw
    MetricTrend data (a chart or dashboard is left to whatever renders
    that data - no UI/plotting dependency is added here).

    Backed by a single JSON array file rather than one file per run:
    simple to append to, trivial to read back in full, and fine at the
    scale a local eval history actually reaches (hundreds of runs, not
    millions of rows).
    """

    def __init__(
        self,
        path: str = DEFAULT_HISTORY_PATH
    ) -> None:
        self.path = Path(path)

    def record(
        self,
        report: EvaluationReport
    ) -> ExperimentRecord:
        entry = ExperimentRecord(
            timestamp=report.metadata.timestamp,
            dataset_name=report.metadata.dataset_name,
            git_commit_hash=report.metadata.git_commit_hash,
            embedding_provider=report.metadata.embedding_provider,
            reranker=report.metadata.reranker,
            generation_provider=report.metadata.generation_provider,
            aggregate_metrics=dict(report.aggregate_metrics)
        )
        history = self._read_all()
        history.append(entry)
        self._write_all(history)
        logger.info(
            "experiment_recorded",
            extra={"dataset": entry.dataset_name, "timestamp": entry.timestamp}
        )
        return entry

    def history(
        self,
        dataset_name: str | None = None,
        limit: int | None = None
    ) -> list[ExperimentRecord]:
        records = self._read_all()

        if dataset_name is not None:
            records = [record for record in records if record.dataset_name == dataset_name]

        if limit is not None:
            records = records[-limit:]

        return records

    def compare_many(
        self,
        reports: list[EvaluationReport]
    ) -> list[MetricTrend]:
        """
        ExperimentTracker Protocol method: builds trends directly from a
        list of already-in-hand EvaluationReports (order = chronological),
        without touching the history file. Use trend_from_history() to
        trend against what's actually been recorded instead.
        """
        records = [
            ExperimentRecord(
                timestamp=report.metadata.timestamp,
                dataset_name=report.metadata.dataset_name,
                git_commit_hash=report.metadata.git_commit_hash,
                embedding_provider=report.metadata.embedding_provider,
                reranker=report.metadata.reranker,
                generation_provider=report.metadata.generation_provider,
                aggregate_metrics=dict(report.aggregate_metrics)
            )
            for report in reports
        ]
        return self._trends_from_records(records)

    def trend_from_history(
        self,
        dataset_name: str,
        limit: int | None = None
    ) -> list[MetricTrend]:
        return self._trends_from_records(self.history(dataset_name=dataset_name, limit=limit))

    def _trends_from_records(
        self,
        records: list[ExperimentRecord]
    ) -> list[MetricTrend]:
        metric_names = sorted({
            metric_name
            for record in records
            for metric_name in record.aggregate_metrics
        })

        return [
            MetricTrend(
                metric=metric_name,
                points=[
                    MetricTrendPoint(
                        timestamp=record.timestamp,
                        value=record.aggregate_metrics[metric_name],
                        git_commit_hash=record.git_commit_hash
                    )
                    for record in records
                    if metric_name in record.aggregate_metrics
                ]
            )
            for metric_name in metric_names
        ]

    def _read_all(self) -> list[ExperimentRecord]:
        if not self.path.exists():
            return []

        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return [ExperimentRecord(**entry) for entry in raw]

    def _write_all(
        self,
        records: list[ExperimentRecord]
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8"
        )


def render_trend_table(
    trends: list[MetricTrend],
    metric_names: list[str] | None = None
) -> str:
    selected = (
        [trend for trend in trends if trend.metric in metric_names]
        if metric_names else trends
    )
    selected = [trend for trend in selected if trend.points]

    if not selected:
        return "No experiment history to trend."

    lines = ["=" * 65]
    lines.append(f"{'Metric':<28}{'Runs':>6}{'Latest':>12}{'Change':>12}")
    lines.append("-" * 65)

    for trend in selected:
        delta = trend.delta_from_previous
        delta_text = f"{delta:+.4f}" if delta is not None else "n/a"
        lines.append(
            f"{trend.metric:<28}{len(trend.points):>6}{trend.latest:>12.4f}{delta_text:>12}"
        )

    lines.append("=" * 65)
    return "\n".join(lines)
