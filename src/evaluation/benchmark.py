import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone

from evaluation.runner import EvaluationRunner
from evaluation.runner import current_git_commit_hash
from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GoldenDataset
from ingestion.ingestion_pipeline import IngestionPipeline
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.retrieval.reranker import CrossEncoderReranker
from rag.vector_store.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    One retrieval configuration to benchmark. `embedder_name` is
    "hashing" (the deterministic MVP default, no download) or any
    sentence-transformers model name (e.g. "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2") to compare real embedding models -
    downloaded on first use, same as SentenceTransformerEmbedder always
    has.

    Chunk-size caveat: golden dataset relevant_chunk_ids are exact
    "document_id:index" strings tied to the chunking parameters the
    dataset was built with. Changing chunk_size/chunk_overlap changes
    every chunk's index, so recall/precision against an unchanged golden
    dataset will legitimately collapse toward zero - that's not a bug in
    the benchmark, it's a limitation of ID-based relevance judgments.
    This config dimension is meaningful for comparing embedder/reranker/
    hybrid-toggle choices at a FIXED chunk size; comparing chunk sizes
    themselves needs a golden dataset rebuilt (or judged by chunk text
    content rather than id) for each size under test.
    """
    label: str
    chunk_size: int = 900
    chunk_overlap: int = 120
    minimum_chunk_size: int = 80
    embedder_name: str = "hashing"
    use_reranker: bool = False
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_multiplier: int = 4
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


class BenchmarkRunner:
    """
    Sweeps a list of BenchmarkConfigs over the same GoldenDataset,
    building a fresh ingestion/retrieval pipeline per config so results
    are never cross-contaminated between runs, and returns one
    EvaluationReport per config for side-by-side comparison.
    """

    def __init__(
        self,
        dataset: GoldenDataset
    ) -> None:
        self.dataset = dataset

    def run(
        self,
        configs: list[BenchmarkConfig]
    ) -> list[tuple[BenchmarkConfig, EvaluationReport]]:
        results = []

        for config in configs:
            logger.info("benchmark_config_started", extra={"label": config.label})
            report = self._run_single(config)
            results.append((config, report))
            logger.info("benchmark_config_completed", extra={"label": config.label})

        return results

    def _run_single(
        self,
        config: BenchmarkConfig
    ) -> EvaluationReport:
        chunker = RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            minimum_chunk_size=config.minimum_chunk_size
        )
        embedder = self._build_embedder(config.embedder_name)
        vector_store = InMemoryVectorStore()
        ingestion_pipeline = IngestionPipeline()

        for file_path in self.dataset.source_documents:
            document_result = ingestion_pipeline.ingest_file(file_path)

            if not document_result.success or document_result.data is None:
                raise ValueError(
                    f"benchmark config '{config.label}': failed to ingest "
                    f"{file_path}: {document_result.error}"
                )

            chunk_result = chunker.chunk(document_result.data)

            if not chunk_result.success or chunk_result.data is None:
                raise ValueError(
                    f"benchmark config '{config.label}': failed to chunk "
                    f"{file_path}: {chunk_result.error}"
                )

            records = [
                (chunk, embedder.embed(chunk.text))
                for chunk in chunk_result.data
            ]
            vector_store.add_many(records)

        retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
        reranker = (
            CrossEncoderReranker(model_name=config.reranker_model_name)
            if config.use_reranker else None
        )

        runner = EvaluationRunner(
            retrieve_fn=self._build_retrieve_fn(retriever, reranker, config),
            k_values=config.k_values
        )
        metadata = ExperimentMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=self.dataset.name,
            embedding_provider=config.embedder_name,
            embedding_model_name=(
                config.embedder_name if config.embedder_name != "hashing" else None
            ),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            retriever="hybrid",
            reranker=config.reranker_model_name if config.use_reranker else None,
            generation_provider=None,
            guardrails_enabled=False,
            git_commit_hash=current_git_commit_hash()
        )

        return runner.run(self.dataset, metadata)

    def _build_retrieve_fn(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None,
        config: BenchmarkConfig
    ):
        def retrieve_fn(query: str, top_k: int) -> list[str]:
            if reranker is None:
                results = retriever.retrieve(query=query, top_k=top_k)
                return [item.chunk.chunk_id for item in results]

            candidates = retriever.retrieve(
                query=query,
                top_k=top_k * config.candidate_multiplier
            )
            reranked = reranker.rerank(query=query, candidates=candidates, top_k=top_k)
            return [item.chunk.chunk_id for item in reranked]

        return retrieve_fn

    def _build_embedder(
        self,
        embedder_name: str
    ):
        if embedder_name == "hashing":
            return HashingEmbedder()

        return SentenceTransformerEmbedder(model_name=embedder_name)


def render_comparison_table(
    results: list[tuple[BenchmarkConfig, EvaluationReport]]
) -> str:
    if not results:
        return "No benchmark results."

    _, first_report = results[0]
    metric_columns = [f"recall@{k}" for k in first_report.k_values] + ["mrr"]

    header = f"{'Config':<24}" + "".join(f"{column:>12}" for column in metric_columns) + f"{'Avg ms':>10}"
    lines = [header, "-" * len(header)]

    for config, report in results:
        row = f"{config.label:<24}"
        row += "".join(f"{report.aggregate_metrics[column]:>12.4f}" for column in metric_columns)
        row += f"{report.retrieval_latency.average_seconds * 1000:>10.1f}"
        lines.append(row)

    return "\n".join(lines)
