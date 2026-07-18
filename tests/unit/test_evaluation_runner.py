from datetime import datetime
from datetime import timezone

from evaluation.runner import EvaluationRunner
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GoldenDataset
from evaluation.schemas import GoldenQuery


def _metadata(dataset_name: str = "test") -> ExperimentMetadata:
    return ExperimentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name=dataset_name,
        embedding_provider="hashing",
        embedding_model_name=None,
        chunk_size=900,
        chunk_overlap=120,
        retriever="hybrid",
        reranker=None,
        generation_provider=None,
        guardrails_enabled=True,
        git_commit_hash=None
    )


def test_runner_computes_metrics_for_each_k_value():

    dataset = GoldenDataset(
        name="test",
        queries=[
            GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"])
        ]
    )

    def retrieve_fn(query, top_k):
        return ["a", "b", "c"][:top_k]

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1, 3])
    report = runner.run(dataset, _metadata())

    assert report.aggregate_metrics["recall@1"] == 1.0
    assert report.aggregate_metrics["recall@3"] == 1.0
    assert report.aggregate_metrics["precision@1"] == 1.0
    assert report.aggregate_metrics["precision@3"] == 1.0 / 3
    assert report.aggregate_metrics["hit_rate@1"] == 1.0
    assert report.aggregate_metrics["mrr"] == 1.0


def test_runner_handles_empty_retrieval_results():

    dataset = GoldenDataset(
        name="test",
        queries=[GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"])]
    )

    def retrieve_fn(query, top_k):
        return []

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1, 5])
    report = runner.run(dataset, _metadata())

    assert report.aggregate_metrics["recall@5"] == 0.0
    assert report.aggregate_metrics["mrr"] == 0.0
    assert report.aggregate_metrics["average_rank"] == 0.0


def test_runner_produces_one_query_evaluation_per_query():

    dataset = GoldenDataset(
        name="test",
        queries=[
            GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"], category="c1", difficulty="easy"),
            GoldenQuery(id="Q2", query="q2", relevant_chunk_ids=["b"], category="c2", difficulty="hard")
        ]
    )

    def retrieve_fn(query, top_k):
        return ["a"] if query == "q1" else ["z"]

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1])
    report = runner.run(dataset, _metadata())

    assert len(report.query_evaluations) == 2
    q1 = next(qe for qe in report.query_evaluations if qe.query_id == "Q1")
    q2 = next(qe for qe in report.query_evaluations if qe.query_id == "Q2")
    assert q1.metrics["recall@1"] == 1.0
    assert q1.category == "c1"
    assert q1.difficulty == "easy"
    assert q2.metrics["recall@1"] == 0.0


def test_runner_computes_latency_stats():

    dataset = GoldenDataset(
        name="test",
        queries=[
            GoldenQuery(id=f"Q{i}", query=f"q{i}", relevant_chunk_ids=["a"])
            for i in range(5)
        ]
    )

    def retrieve_fn(query, top_k):
        return ["a"]

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1])
    report = runner.run(dataset, _metadata())

    assert report.retrieval_latency.average_seconds >= 0.0
    assert report.retrieval_latency.median_seconds >= 0.0
    assert report.retrieval_latency.p95_seconds >= 0.0


def test_runner_uses_max_k_for_single_retrieve_call():

    calls = []

    dataset = GoldenDataset(
        name="test",
        queries=[GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"])]
    )

    def retrieve_fn(query, top_k):
        calls.append(top_k)
        return ["a"] * top_k

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1, 3, 5, 10])
    runner.run(dataset, _metadata())

    # only one retrieve() call per query, sized to the largest k so every
    # smaller k can be sliced from the same result set
    assert calls == [10]


def test_runner_metadata_is_carried_through_to_report():

    dataset = GoldenDataset(
        name="test",
        queries=[GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"])]
    )
    metadata = _metadata(dataset_name="carried-through")

    runner = EvaluationRunner(retrieve_fn=lambda query, top_k: ["a"], k_values=[1])
    report = runner.run(dataset, metadata)

    assert report.metadata is metadata
    assert report.metadata.dataset_name == "carried-through"
