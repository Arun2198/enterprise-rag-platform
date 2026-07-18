from evaluation.benchmark import BenchmarkConfig
from evaluation.benchmark import BenchmarkRunner
from evaluation.dataset import load_dataset
from evaluation.report import render_console
from evaluation.report import write_csv_report
from evaluation.report import write_json_report


def test_golden_dataset_to_report_end_to_end(tmp_path):
    """
    Golden Dataset -> Retriever -> Metrics -> Report, using the real
    sample_documents/AI-RMF-1stdraft.pdf and the real golden_dataset.json
    (hashing embedder, no reranker - fast, no downloads). Also guards the
    dataset itself: if RecursiveChunker's defaults ever change, this
    starts failing and signals the golden dataset needs rebuilding.
    """
    dataset = load_dataset("evaluation/golden_dataset.json")
    runner = BenchmarkRunner(dataset)

    config = BenchmarkConfig(label="integration", k_values=[1, 3, 5, 10])
    ((_, report),) = runner.run([config])

    assert len(report.query_evaluations) == 20
    # every relevant_chunk_id must exist in the real chunked corpus - if
    # the dataset is stale (rebuilt document, changed chunker defaults)
    # every query would silently score zero instead of failing loudly
    assert report.aggregate_metrics["recall@10"] > 0.5
    assert report.aggregate_metrics["mrr"] > 0.0

    console_text = render_console(report)
    assert "ai-rmf-1stdraft" in console_text

    json_path = write_json_report(report, str(tmp_path))
    csv_path = write_csv_report(report, str(tmp_path))
    assert json_path.exists()
    assert csv_path.exists()
