"""
Phase 31 - provider benchmarking.

Compares candidate embedding and reranking implementations (local vs.
API-based) against the real golden dataset with actual measured
results - the spec is explicit: "Do not claim one is superior without
measured results." Two isolated comparisons, run separately so one
dimension's effect isn't conflated with the other's:

  A. Embedding providers (reranker off): hashing (MVP baseline,
     deterministic, no download) vs. a real local dense model
     (BAAI/bge-small-en-v1.5) vs. Jina (jina-embeddings-v3, real API)
  B. Reranker providers (embedder fixed to the real local dense model
     from A, so the comparison isolates the reranking stage): none vs.
     local cross-encoder (ms-marco-MiniLM-L-6-v2) vs. Jina
     (jina-reranker-v2-base-multilingual, real API)

Jina legs only run when JINA_API_KEY is set in the environment - never
required, never fabricated when absent.

Uses RecursiveChunker's default chunk_size/chunk_overlap/
minimum_chunk_size (900/120/80) since evaluation/golden_dataset.json's
relevant_chunk_ids are positional and only valid at those exact
chunking parameters (see BenchmarkConfig's own docstring).

Run:
    set -a; source .env; set +a
    uv run python evaluation/run_provider_benchmark.py
"""
import logging
import os
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation import report  # noqa: E402
from evaluation.benchmark import BenchmarkConfig  # noqa: E402
from evaluation.benchmark import BenchmarkRunner  # noqa: E402
from evaluation.benchmark import render_comparison_table  # noqa: E402
from evaluation.dataset import load_dataset  # noqa: E402
from evaluation.schemas import EvaluationReport  # noqa: E402

K_VALUES = [1, 3, 5, 10]
LOCAL_DENSE_MODEL = "BAAI/bge-small-en-v1.5"


def main() -> int:
    # INFO, not WARNING - BenchmarkRunner.run() already logs
    # benchmark_config_started/completed per config, which is the only
    # real-time progress signal available while a config is running
    # (ingestion + embedding of ~148 chunks per config has no finer-
    # grained progress hook without changing the library itself).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s %(extra)s" if False else "%(asctime)s %(message)s")
    jina_key = os.getenv("JINA_API_KEY")
    dataset = load_dataset("evaluation/golden_dataset.json")
    runner = BenchmarkRunner(dataset)
    output_dir = "evaluation/reports"
    all_results: list[tuple[BenchmarkConfig, EvaluationReport]] = []

    print(f"Dataset: {dataset.name} ({len(dataset.queries)} queries)")
    print(f"Jina API key present: {bool(jina_key)}\n")

    # --- Comparison A: embedding providers, reranker off -----------
    embedding_configs = [
        BenchmarkConfig(
            label="embed:hashing",
            embedder_provider="local", embedder_name="hashing",
            k_values=K_VALUES
        ),
        BenchmarkConfig(
            label=f"embed:local:{LOCAL_DENSE_MODEL}",
            embedder_provider="local", embedder_name=LOCAL_DENSE_MODEL,
            k_values=K_VALUES
        ),
    ]

    if jina_key:
        embedding_configs.append(
            BenchmarkConfig(
                label="embed:jina:jina-embeddings-v3",
                embedder_provider="jina", embedder_api_key=jina_key,
                k_values=K_VALUES
            )
        )
    else:
        print("JINA_API_KEY not set - skipping the Jina embedding leg.", file=sys.stderr)

    print("=== Comparison A: embedding providers (no reranker) ===")
    embedding_results = runner.run(embedding_configs)
    print(render_comparison_table(embedding_results))
    print()
    all_results.extend(embedding_results)

    # --- Comparison B: reranker providers, embedder fixed to the ---
    # real local dense model (isolates the reranking stage from the
    # embedding-quality confound hashing would introduce)
    reranker_configs = [
        BenchmarkConfig(
            label="rerank:none",
            embedder_provider="local", embedder_name=LOCAL_DENSE_MODEL,
            reranker_provider="none",
            k_values=K_VALUES
        ),
        BenchmarkConfig(
            label="rerank:local:ms-marco-MiniLM-L-6-v2",
            embedder_provider="local", embedder_name=LOCAL_DENSE_MODEL,
            reranker_provider="local",
            k_values=K_VALUES
        ),
    ]

    if jina_key:
        reranker_configs.append(
            BenchmarkConfig(
                label="rerank:jina:jina-reranker-v2-base-multilingual",
                embedder_provider="local", embedder_name=LOCAL_DENSE_MODEL,
                reranker_provider="jina", reranker_api_key=jina_key,
                k_values=K_VALUES
            )
        )
    else:
        print("JINA_API_KEY not set - skipping the Jina reranker leg.", file=sys.stderr)

    print("=== Comparison B: reranker providers (embedder fixed) ===")
    reranker_results = runner.run(reranker_configs)
    print(render_comparison_table(reranker_results))
    print()
    all_results.extend(reranker_results)

    # --- Persist every leg as its own JSON report, same shape as ---
    # run_eval.py's --json output, so results are reproducible and
    # inspectable later, not just printed once and lost.
    for _config, evaluation_report in all_results:
        path = report.write_json_report(evaluation_report, output_dir)
        print(f"wrote {path}")

    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"\nBenchmark completed at {timestamp}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
