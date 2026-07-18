from evaluation.benchmark import BenchmarkConfig
from evaluation.benchmark import BenchmarkRunner
from evaluation.benchmark import render_comparison_table
from evaluation.dataset import load_dataset
from ingestion.ingestion_pipeline import IngestionPipeline
from rag.chunking.recursive_chunker import RecursiveChunker


def _build_dataset_file(tmp_path):
    doc_path = tmp_path / "policy.md"
    doc_path.write_text(
        "# Leave Policy\n"
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave. "
        "All requests must be submitted two weeks in advance.",
        encoding="utf-8"
    )

    # discover real chunk ids the same way the real golden dataset was
    # built, so the fixture is grounded rather than guessed
    document = IngestionPipeline().ingest_file(str(doc_path)).data
    chunks = RecursiveChunker(chunk_size=60, chunk_overlap=10, minimum_chunk_size=5).chunk(document).data
    contractor_chunk = next(c for c in chunks if "Contractors" in c.text)

    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        f'{{"name": "benchmark-fixture", "source_documents": ["{str(doc_path).replace(chr(92), "/")}"], '
        f'"queries": [{{"id": "Q1", "query": "How many leave days do contractors receive?", '
        f'"relevant_chunk_ids": ["{contractor_chunk.chunk_id}"]}}]}}',
        encoding="utf-8"
    )
    return str(dataset_path)


def test_benchmark_runner_produces_one_report_per_config(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    configs = [
        BenchmarkConfig(label="chunk_60", chunk_size=60, chunk_overlap=10, minimum_chunk_size=5, k_values=[1, 3]),
        BenchmarkConfig(label="chunk_30", chunk_size=30, chunk_overlap=5, minimum_chunk_size=5, k_values=[1, 3])
    ]
    results = runner.run(configs)

    assert len(results) == 2
    labels = [config.label for config, _ in results]
    assert labels == ["chunk_60", "chunk_30"]


def test_benchmark_runner_matching_config_finds_the_relevant_chunk(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    # same chunk_size/overlap the fixture's relevant_chunk_ids were built
    # with, so this config's ids are actually valid against its own chunks
    config = BenchmarkConfig(
        label="matching",
        chunk_size=60,
        chunk_overlap=10,
        minimum_chunk_size=5,
        k_values=[1, 3]
    )
    ((_, evaluation_report),) = runner.run([config])

    assert evaluation_report.aggregate_metrics["recall@3"] == 1.0


def test_benchmark_runner_reranker_toggle_does_not_crash(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    config = BenchmarkConfig(
        label="with_reranker",
        chunk_size=60,
        chunk_overlap=10,
        minimum_chunk_size=5,
        use_reranker=True,
        k_values=[1, 3]
    )
    results = runner.run([config])

    assert len(results) == 1


def test_render_comparison_table_includes_all_configs(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    configs = [
        BenchmarkConfig(label="config_a", chunk_size=60, chunk_overlap=10, minimum_chunk_size=5, k_values=[1, 3]),
        BenchmarkConfig(label="config_b", chunk_size=30, chunk_overlap=5, minimum_chunk_size=5, k_values=[1, 3])
    ]
    results = runner.run(configs)

    table = render_comparison_table(results)

    assert "config_a" in table
    assert "config_b" in table
    assert "recall@1" in table


def test_render_comparison_table_handles_empty_results():

    assert render_comparison_table([]) == "No benchmark results."
