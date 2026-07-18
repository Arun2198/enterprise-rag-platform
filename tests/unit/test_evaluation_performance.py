import time

from evaluation.dataset import load_dataset
from evaluation.runner import EvaluationRunner
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GoldenDataset
from evaluation.schemas import GoldenQuery
from datetime import datetime
from datetime import timezone

ITERATIONS = 500


def _metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_name="perf-test",
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


def test_dataset_loading_is_fast():

    started_at = time.perf_counter()

    for _ in range(50):
        load_dataset("evaluation/golden_dataset.json")

    elapsed = time.perf_counter() - started_at

    assert elapsed / 50 < 0.05


def test_runner_throughput_over_many_queries():

    dataset = GoldenDataset(
        name="perf-test",
        queries=[
            GoldenQuery(id=f"Q{i}", query=f"query {i}", relevant_chunk_ids=["a"])
            for i in range(ITERATIONS)
        ]
    )

    def retrieve_fn(query, top_k):
        return ["a", "b", "c"][:top_k]

    runner = EvaluationRunner(retrieve_fn=retrieve_fn, k_values=[1, 5, 10])

    started_at = time.perf_counter()
    report = runner.run(dataset, _metadata())
    elapsed = time.perf_counter() - started_at

    throughput_per_second = ITERATIONS / max(elapsed, 1e-9)

    assert len(report.query_evaluations) == ITERATIONS
    assert throughput_per_second > 500


def test_runner_latency_overhead_per_query_is_small():

    dataset = GoldenDataset(
        name="perf-test",
        queries=[GoldenQuery(id="Q1", query="q1", relevant_chunk_ids=["a"])]
    )

    def instant_retrieve(query, top_k):
        return ["a"]

    runner = EvaluationRunner(retrieve_fn=instant_retrieve, k_values=[1, 5, 10])
    report = runner.run(dataset, _metadata())

    # the runner's own bookkeeping (metric computation, dataclass
    # construction) shouldn't dominate over an instant retrieve_fn
    assert report.retrieval_latency.average_seconds < 0.01
