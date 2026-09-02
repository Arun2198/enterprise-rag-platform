from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from evaluation.benchmark import BenchmarkConfig
from evaluation.benchmark import BenchmarkRunner
from evaluation.benchmark import render_comparison_table
from evaluation.dataset import load_dataset
from ingestion.ingestion_pipeline import IngestionPipeline
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.embeddings.jina_embedder import JinaEmbedder
from rag.retrieval.cohere_reranker import CohereReranker
from rag.retrieval.jina_reranker import JinaReranker


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
    document = IngestionPipeline().ingest_file(str(doc_path), document_id="policy").data
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


def test_benchmark_runner_with_generation_populates_answers_and_generation_metrics(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    config = BenchmarkConfig(
        label="with_generation",
        chunk_size=60,
        chunk_overlap=10,
        minimum_chunk_size=5,
        k_values=[1, 3],
        generation_provider="extractive"
    )
    ((_, evaluation_report),) = runner.run([config])

    query_evaluation = evaluation_report.query_evaluations[0]
    assert query_evaluation.answer is not None
    assert "groundedness" in query_evaluation.generation_metrics
    assert "answer_relevance" in query_evaluation.generation_metrics
    assert "context_relevance" in query_evaluation.generation_metrics
    assert "generation/groundedness" in evaluation_report.aggregate_metrics
    assert evaluation_report.metadata.generation_provider == "extractive"


def test_benchmark_runner_openai_compatible_generation_requires_credentials(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)

    config = BenchmarkConfig(
        label="missing_creds",
        chunk_size=60,
        chunk_overlap=10,
        minimum_chunk_size=5,
        k_values=[1, 3],
        generation_provider="openai_compatible"
    )

    try:
        runner.run([config])
        raise AssertionError("expected ValueError")
    except ValueError as ex:
        assert "llm_base_url" in str(ex)


def _fake_jina_embedding_post(url, json=None, headers=None, timeout=None):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "data": [{"index": i, "embedding": [0.1] * 1024} for i in range(len(json["input"]))]
    }
    return response


def _fake_cohere_embedding_post(url, json=None, headers=None, timeout=None):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "embeddings": {"float": [[0.1] * 1024 for _ in json["texts"]]}
    }
    return response


def _fake_rerank_post(url, json=None, headers=None, timeout=None):
    # scores documents by whether they mention "Contractors" so the
    # benchmark's real relevance judgment (recall@k) is meaningfully
    # exercised, not just "does this call not crash"
    response = MagicMock()
    response.raise_for_status = MagicMock()
    documents = json["documents"]
    scored = sorted(
        range(len(documents)),
        key=lambda i: 0 if "Contractors" in documents[i] else 1
    )
    response.json.return_value = {
        "results": [
            {"index": i, "relevance_score": 1.0 - (rank * 0.1)}
            for rank, i in enumerate(scored[:json["top_n"]])
        ]
    }
    return response


def _api_provider_config(**overrides):
    defaults = dict(
        label="api_provider",
        chunk_size=60,
        chunk_overlap=10,
        minimum_chunk_size=5,
        k_values=[1, 3]
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


@patch("rag.embeddings.jina_embedder.requests.post", side_effect=_fake_jina_embedding_post)
def test_benchmark_runner_builds_jina_embedder_when_configured(mock_post, tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(embedder_provider="jina", embedder_api_key="fake-key")

    ((used_config, evaluation_report),) = runner.run([config])

    assert mock_post.called
    assert evaluation_report.metadata.embedding_provider == "jina"
    assert evaluation_report.metadata.embedding_model_name == "jina-embeddings-v3"
    assert used_config.label == "api_provider"


@patch("rag.embeddings.cohere_embedder.requests.post", side_effect=_fake_cohere_embedding_post)
def test_benchmark_runner_builds_cohere_embedder_when_configured(mock_post, tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(embedder_provider="cohere", embedder_api_key="fake-key")

    ((_, evaluation_report),) = runner.run([config])

    assert mock_post.called
    assert evaluation_report.metadata.embedding_provider == "cohere"


def test_benchmark_runner_jina_embedder_without_api_key_raises(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(embedder_provider="jina")

    with pytest.raises(ValueError, match="embedder_api_key"):
        runner.run([config])


def test_benchmark_runner_unknown_embedder_provider_raises(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(embedder_provider="not-a-real-provider")

    with pytest.raises(ValueError, match="embedder_provider"):
        runner.run([config])


@patch("rag.retrieval.jina_reranker.requests.post", side_effect=_fake_rerank_post)
def test_benchmark_runner_builds_jina_reranker_when_configured(mock_post, tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(reranker_provider="jina", reranker_api_key="fake-key")

    ((_, evaluation_report),) = runner.run([config])

    assert mock_post.called
    assert evaluation_report.metadata.reranker == "jina:jina-reranker-v2-base-multilingual"
    assert evaluation_report.aggregate_metrics["recall@3"] == 1.0


@patch("rag.retrieval.cohere_reranker.requests.post", side_effect=_fake_rerank_post)
def test_benchmark_runner_builds_cohere_reranker_when_configured(mock_post, tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(reranker_provider="cohere", reranker_api_key="fake-key")

    ((_, evaluation_report),) = runner.run([config])

    assert mock_post.called
    assert evaluation_report.metadata.reranker == "cohere:rerank-english-v3.0"


def test_benchmark_runner_reranker_provider_none_overrides_use_reranker_true(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    # reranker_provider left unset (None) means "derive from use_reranker" -
    # explicitly proving the back-compat path still resolves to "local"
    config = _api_provider_config(use_reranker=True)

    ((_, evaluation_report),) = runner.run([config])

    assert evaluation_report.metadata.reranker is not None
    assert evaluation_report.metadata.reranker.startswith("local:")


def test_benchmark_runner_reranker_provider_none_string_disables_reranking(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    # explicit reranker_provider="none" disables reranking even if
    # use_reranker=True, since an explicit provider always wins
    config = _api_provider_config(use_reranker=True, reranker_provider="none")

    ((_, evaluation_report),) = runner.run([config])

    assert evaluation_report.metadata.reranker is None


def test_benchmark_runner_unknown_reranker_provider_raises(tmp_path):

    dataset = load_dataset(_build_dataset_file(tmp_path))
    runner = BenchmarkRunner(dataset)
    config = _api_provider_config(reranker_provider="not-a-real-provider")

    with pytest.raises(ValueError, match="reranker_provider"):
        runner.run([config])
