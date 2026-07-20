from datetime import datetime
from datetime import timezone

from evaluation import report
from evaluation.runner import EvaluationRunner
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GoldenDataset
from evaluation.schemas import GoldenQuery


def _metadata(**overrides) -> ExperimentMetadata:
    defaults = dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name="test-dataset",
        embedding_provider="hashing",
        embedding_model_name=None,
        chunk_size=900,
        chunk_overlap=120,
        retriever="hybrid",
        reranker=None,
        generation_provider=None,
        guardrails_enabled=True,
        git_commit_hash="abc1234"
    )
    defaults.update(overrides)
    return ExperimentMetadata(**defaults)


def _build_report(
    retrieve_fn,
    k_values=(1, 5),
    dataset_name="test-dataset",
    answer_fn=None,
    generation_metrics=None
):
    dataset = GoldenDataset(
        name=dataset_name,
        queries=[
            GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"], category="c", difficulty="easy"),
            GoldenQuery(id="Q2", query="q2", relevant_chunk_ids=["b"], category="c", difficulty="hard")
        ]
    )
    runner = EvaluationRunner(
        retrieve_fn=retrieve_fn,
        k_values=list(k_values),
        answer_fn=answer_fn,
        generation_metrics=generation_metrics
    )
    return runner.run(dataset, _metadata(dataset_name=dataset_name))


class _FixedScoreMetric:
    name = "fixed_metric"

    def score(self, query, answer, retrieved_chunk_texts):
        return 0.75


def test_render_console_contains_key_sections():

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    text = report.render_console(evaluation_report)

    assert "test-dataset" in text
    assert "Recall@1" in text
    assert "Recall@5" in text
    assert "MRR" in text
    assert "Hit Rate@5" in text
    assert "NDCG@5" in text
    assert "Average Rank" in text
    assert "Average Retrieval Time" in text


def test_json_round_trip_preserves_report(tmp_path):

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    written_path = report.write_json_report(evaluation_report, str(tmp_path))
    loaded = report.load_json_report(str(written_path))

    assert loaded == evaluation_report


def test_write_json_report_creates_timestamped_file(tmp_path):

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    path = report.write_json_report(evaluation_report, str(tmp_path))

    assert path.exists()
    assert path.suffix == ".json"
    assert "test-dataset" in path.name


def test_render_csv_has_one_row_per_query_plus_header():

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    csv_text = report.render_csv(evaluation_report)
    lines = csv_text.strip().splitlines()

    assert len(lines) == 3  # header + 2 queries
    assert "query_id" in lines[0]
    assert "Q1" in lines[1]
    assert "Q2" in lines[2]


def test_write_csv_report_creates_file(tmp_path):

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    path = report.write_csv_report(evaluation_report, str(tmp_path))

    assert path.exists()
    assert path.suffix == ".csv"


def test_compare_reports_detects_regression():

    baseline = _build_report(lambda query, top_k: ["a"] if query == "q1" else ["b"])
    current = _build_report(lambda query, top_k: [])

    comparison = report.compare_reports(current=current, baseline=baseline, threshold=0.02)

    assert comparison.has_regressions is True
    recall_delta = next(d for d in comparison.deltas if d.metric == "recall@1")
    assert recall_delta.is_regression is True
    assert recall_delta.current == 0.0
    assert recall_delta.baseline == 1.0


def test_compare_reports_no_regression_when_identical():

    baseline = _build_report(lambda query, top_k: ["a"] if query == "q1" else ["b"])
    current = _build_report(lambda query, top_k: ["a"] if query == "q1" else ["b"])

    comparison = report.compare_reports(current=current, baseline=baseline, threshold=0.02)

    assert comparison.has_regressions is False


def test_compare_reports_ignores_metrics_missing_from_baseline():

    baseline = _build_report(lambda query, top_k: ["a"], k_values=(1,))
    current = _build_report(lambda query, top_k: ["a"], k_values=(1, 5))

    comparison = report.compare_reports(current=current, baseline=baseline)

    metric_names = {delta.metric for delta in comparison.deltas}
    assert "recall@5" not in metric_names
    assert "recall@1" in metric_names


def test_render_console_includes_generation_section_when_present():

    evaluation_report = _build_report(
        lambda query, top_k: ["a"],
        answer_fn=lambda query, ids: ("an answer", ["context"]),
        generation_metrics=[_FixedScoreMetric()]
    )

    text = report.render_console(evaluation_report)

    assert "Generation quality (Layer 2)" in text
    assert "fixed_metric: 0.7500" in text


def test_render_console_omits_generation_section_when_absent():

    evaluation_report = _build_report(lambda query, top_k: ["a"])

    text = report.render_console(evaluation_report)

    assert "Generation quality" not in text


def test_render_csv_includes_answer_and_generation_columns():

    evaluation_report = _build_report(
        lambda query, top_k: ["a"],
        answer_fn=lambda query, ids: ("an answer", ["context"]),
        generation_metrics=[_FixedScoreMetric()]
    )

    csv_text = report.render_csv(evaluation_report)
    lines = csv_text.strip().splitlines()

    assert "answer" in lines[0]
    assert "fixed_metric" in lines[0]
    assert "an answer" in lines[1]


def test_json_round_trip_preserves_generation_fields(tmp_path):

    evaluation_report = _build_report(
        lambda query, top_k: ["a"],
        answer_fn=lambda query, ids: ("an answer", ["context"]),
        generation_metrics=[_FixedScoreMetric()]
    )

    written_path = report.write_json_report(evaluation_report, str(tmp_path))
    loaded = report.load_json_report(str(written_path))

    assert loaded == evaluation_report
    assert loaded.query_evaluations[0].answer == "an answer"
    assert loaded.query_evaluations[0].generation_metrics["fixed_metric"] == 0.75


def test_render_system_metrics_formats_sorted_lines():

    text = report.render_system_metrics({"queries_evaluated": 3.0, "retrieval_throughput_qps": 12.5})

    assert "System metrics (Layer 3)" in text
    assert "queries_evaluated: 3.0000" in text
    assert "retrieval_throughput_qps: 12.5000" in text


def test_render_system_metrics_handles_empty_dict():

    assert report.render_system_metrics({}) == ""


def test_render_comparison_console_flags_regressions():

    baseline = _build_report(lambda query, top_k: ["a"] if query == "q1" else ["b"])
    current = _build_report(lambda query, top_k: [])

    comparison = report.compare_reports(current=current, baseline=baseline, threshold=0.02)
    text = report.render_comparison_console(comparison)

    assert "REGRESSION" in text
    assert "REGRESSIONS DETECTED" in text
