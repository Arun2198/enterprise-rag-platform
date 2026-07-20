from datetime import datetime
from datetime import timezone

from evaluation.experiment_tracker import LocalExperimentTracker
from evaluation.experiment_tracker import render_trend_table
from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import LatencyStats


def _report(
    dataset_name: str = "test",
    git_commit_hash: str | None = "abc123",
    **metric_overrides
) -> EvaluationReport:
    return EvaluationReport(
        metadata=ExperimentMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=dataset_name,
            embedding_provider="hashing",
            embedding_model_name=None,
            chunk_size=900,
            chunk_overlap=120,
            retriever="hybrid",
            reranker=None,
            generation_provider=None,
            guardrails_enabled=False,
            git_commit_hash=git_commit_hash
        ),
        k_values=[1],
        aggregate_metrics={"recall@1": 0.5, **metric_overrides},
        query_evaluations=[],
        retrieval_latency=LatencyStats(average_seconds=0.0, median_seconds=0.0, p95_seconds=0.0)
    )


def test_record_appends_to_a_fresh_history_file(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))

    tracker.record(_report())

    history = tracker.history()
    assert len(history) == 1
    assert history[0].aggregate_metrics["recall@1"] == 0.5


def test_record_is_append_only_across_multiple_runs(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))

    tracker.record(_report(**{"recall@1": 0.5}))
    tracker.record(_report(**{"recall@1": 0.7}))

    history = tracker.history()
    assert len(history) == 2
    assert [r.aggregate_metrics["recall@1"] for r in history] == [0.5, 0.7]


def test_history_filters_by_dataset_name(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))
    tracker.record(_report(dataset_name="a"))
    tracker.record(_report(dataset_name="b"))

    assert len(tracker.history(dataset_name="a")) == 1
    assert len(tracker.history(dataset_name="b")) == 1


def test_history_limit_returns_most_recent_entries(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))

    for i in range(5):
        tracker.record(_report(**{"recall@1": float(i)}))

    limited = tracker.history(limit=2)
    assert [r.aggregate_metrics["recall@1"] for r in limited] == [3.0, 4.0]


def test_trend_from_history_tracks_a_metric_across_runs(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))
    tracker.record(_report(**{"recall@1": 0.4}))
    tracker.record(_report(**{"recall@1": 0.6}))

    trends = tracker.trend_from_history("test")
    recall_trend = next(t for t in trends if t.metric == "recall@1")

    assert [p.value for p in recall_trend.points] == [0.4, 0.6]
    assert recall_trend.latest == 0.6
    assert recall_trend.delta_from_previous == 0.6 - 0.4


def test_compare_many_builds_trends_without_touching_history_file(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))
    reports = [_report(**{"recall@1": 0.3}), _report(**{"recall@1": 0.9})]

    trends = tracker.compare_many(reports)

    assert tracker.history() == []  # compare_many is read-only w.r.t. the history file
    recall_trend = next(t for t in trends if t.metric == "recall@1")
    assert [p.value for p in recall_trend.points] == [0.3, 0.9]


def test_metric_trend_delta_from_previous_is_none_with_fewer_than_two_points(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))
    tracker.record(_report())

    trends = tracker.trend_from_history("test")
    recall_trend = next(t for t in trends if t.metric == "recall@1")

    assert recall_trend.delta_from_previous is None


def test_render_trend_table_includes_metric_and_latest_value(tmp_path):

    tracker = LocalExperimentTracker(path=str(tmp_path / "history.json"))
    tracker.record(_report(**{"recall@1": 0.4}))
    tracker.record(_report(**{"recall@1": 0.6}))

    text = render_trend_table(tracker.trend_from_history("test"))

    assert "recall@1" in text
    assert "0.6000" in text


def test_render_trend_table_handles_no_history():

    assert render_trend_table([]) == "No experiment history to trend."
