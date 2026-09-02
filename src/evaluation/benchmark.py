import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from evaluation.generation_metrics import AnswerRelevanceMetric
from evaluation.generation_metrics import ContextRelevanceMetric
from evaluation.generation_metrics import GroundednessMetric
from evaluation.generation_metrics import LLMJudgeGenerationMetric
from evaluation.runner import EvaluationRunner
from evaluation.runner import current_git_commit_hash
from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GenerationMetric
from evaluation.schemas import GoldenDataset
from ingestion.ingestion_pipeline import IngestionPipeline
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.base import Embedder
from rag.embeddings.cohere_embedder import CohereEmbedder
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.embeddings.jina_embedder import JinaEmbedder
from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
from rag.generation.base import Answerer
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.retrieval.cohere_reranker import CohereReranker
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.jina_reranker import JinaReranker
from rag.retrieval.reranker import CrossEncoderReranker
from rag.vector_store.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)

GENERATION_PROVIDERS = ("extractive", "openai_compatible")
EMBEDDER_PROVIDERS = ("local", "jina", "cohere")
RERANKER_PROVIDERS = ("none", "local", "jina", "cohere")


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
    generation_provider: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0

    # --- API-based embedding/reranking providers (Phase 31: provider
    # benchmarking) - additive, backward-compatible with the fields
    # above. embedder_provider="local" (the default) preserves the
    # original embedder_name-driven behavior exactly. reranker_provider
    # left as None means "derive from use_reranker", also preserving
    # original behavior exactly - set it explicitly to "jina"/"cohere"
    # to benchmark an API-based reranker instead of (or in addition to
    # comparing against) the local cross-encoder.
    embedder_provider: str = "local"
    embedder_api_key: str | None = None
    embedder_model_name: str | None = None
    reranker_provider: str | None = None
    reranker_api_key: str | None = None
    reranker_model_override: str | None = None


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
        embedder = self._build_embedder(config)
        vector_store = InMemoryVectorStore()
        ingestion_pipeline = IngestionPipeline()

        for file_path in self.dataset.source_documents:
            # Preserves the same document_id a bare filename-derived
            # ingest would have produced (Path(file_path).stem) - golden
            # datasets' relevant_chunk_ids are built against that exact
            # id, so this must stay stable even though document_id is
            # now a mandatory, explicit argument rather than an implicit
            # default.
            document_result = ingestion_pipeline.ingest_file(file_path, document_id=Path(file_path).stem)

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
        effective_reranker_provider = (
            config.reranker_provider
            if config.reranker_provider is not None
            else ("local" if config.use_reranker else "none")
        )
        reranker = self._build_reranker(config, effective_reranker_provider)

        answer_fn = None
        generation_metrics: list[GenerationMetric] = []

        if config.generation_provider is not None:
            answerer = self._build_answerer(config)
            answer_fn = self._build_answer_fn(vector_store, answerer)
            generation_metrics = self._build_generation_metrics(config, embedder)

        runner = EvaluationRunner(
            retrieve_fn=self._build_retrieve_fn(retriever, reranker, config),
            k_values=config.k_values,
            answer_fn=answer_fn,
            generation_metrics=generation_metrics
        )
        metadata = ExperimentMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_name=self.dataset.name,
            embedding_provider=getattr(embedder, "provider_name", config.embedder_provider),
            embedding_model_name=getattr(embedder, "model_name", None),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            retriever="hybrid",
            reranker=(
                f"{effective_reranker_provider}:{reranker.model_name}"
                if reranker is not None else None
            ),
            generation_provider=config.generation_provider,
            guardrails_enabled=False,
            git_commit_hash=current_git_commit_hash()
        )

        return runner.run(self.dataset, metadata)

    def _build_retrieve_fn(
        self,
        retriever: HybridRetriever,
        reranker: Any,
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
        config: BenchmarkConfig
    ) -> Embedder:
        if config.embedder_provider == "jina":
            if not config.embedder_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': embedder_provider=jina "
                    "requires embedder_api_key"
                )

            kwargs: dict[str, Any] = {"api_key": config.embedder_api_key}

            if config.embedder_model_name:
                kwargs["model_name"] = config.embedder_model_name

            return JinaEmbedder(**kwargs)

        if config.embedder_provider == "cohere":
            if not config.embedder_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': embedder_provider=cohere "
                    "requires embedder_api_key"
                )

            kwargs = {"api_key": config.embedder_api_key}

            if config.embedder_model_name:
                kwargs["model_name"] = config.embedder_model_name

            return CohereEmbedder(**kwargs)

        if config.embedder_provider != "local":
            raise ValueError(
                f"benchmark config '{config.label}': embedder_provider must be one "
                f"of {EMBEDDER_PROVIDERS}, got {config.embedder_provider!r}"
            )

        if config.embedder_name == "hashing":
            return HashingEmbedder()

        return SentenceTransformerEmbedder(model_name=config.embedder_name)

    def _build_reranker(
        self,
        config: BenchmarkConfig,
        effective_provider: str
    ) -> Any:
        if effective_provider == "none":
            return None

        if effective_provider == "local":
            return CrossEncoderReranker(model_name=config.reranker_model_name)

        if effective_provider == "jina":
            if not config.reranker_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': reranker_provider=jina "
                    "requires reranker_api_key"
                )

            kwargs: dict[str, Any] = {"api_key": config.reranker_api_key}

            if config.reranker_model_override:
                kwargs["model_name"] = config.reranker_model_override

            return JinaReranker(**kwargs)

        if effective_provider == "cohere":
            if not config.reranker_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': reranker_provider=cohere "
                    "requires reranker_api_key"
                )

            kwargs = {"api_key": config.reranker_api_key}

            if config.reranker_model_override:
                kwargs["model_name"] = config.reranker_model_override

            return CohereReranker(**kwargs)

        raise ValueError(
            f"benchmark config '{config.label}': reranker_provider must be one of "
            f"{RERANKER_PROVIDERS}, got {effective_provider!r}"
        )

    def _build_answerer(
        self,
        config: BenchmarkConfig
    ) -> Answerer:
        if config.generation_provider == "extractive":
            return ExtractiveAnswerer()

        if config.generation_provider == "openai_compatible":
            if not config.llm_base_url or not config.llm_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': generation_provider="
                    "openai_compatible requires llm_base_url and llm_api_key"
                )

            return OpenAICompatibleAnswerer(
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
                model_name=config.llm_model_name,
                timeout=config.llm_timeout_seconds
            )

        raise ValueError(
            f"benchmark config '{config.label}': generation_provider must be one "
            f"of {GENERATION_PROVIDERS}, got {config.generation_provider!r}"
        )

    def _build_answer_fn(
        self,
        vector_store: InMemoryVectorStore,
        answerer: Answerer
    ):
        def answer_fn(query: str, retrieved_ids: list[str]) -> tuple[str, list[str]]:
            retrieved_chunks = [
                RetrievedChunk(chunk=chunk, vector_score=0.0, keyword_score=0.0, score=0.0)
                for chunk in (vector_store.get(chunk_id) for chunk_id in retrieved_ids)
                if chunk is not None
            ]
            answer = answerer.answer(query=query, retrieved_chunks=retrieved_chunks)
            return answer, [item.chunk.text for item in retrieved_chunks]

        return answer_fn

    def _build_generation_metrics(
        self,
        config: BenchmarkConfig,
        embedder: Embedder
    ) -> list[GenerationMetric]:
        generation_metrics: list[GenerationMetric] = [
            GroundednessMetric(embedder=embedder),
            AnswerRelevanceMetric(embedder=embedder),
            ContextRelevanceMetric()
        ]

        if config.generation_provider == "openai_compatible":
            if not config.llm_base_url or not config.llm_api_key:
                raise ValueError(
                    f"benchmark config '{config.label}': generation_provider="
                    "openai_compatible requires llm_base_url and llm_api_key"
                )

            generation_metrics.append(
                LLMJudgeGenerationMetric(
                    api_key=config.llm_api_key,
                    base_url=config.llm_base_url,
                    model_name=config.llm_model_name,
                    timeout=config.llm_timeout_seconds
                )
            )

        return generation_metrics


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
