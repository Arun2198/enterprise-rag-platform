import logging
import math
import statistics
import subprocess
import time
from typing import Callable

from evaluation import metrics
from evaluation.schemas import EvaluationReport
from evaluation.schemas import ExperimentMetadata
from evaluation.schemas import GoldenDataset
from evaluation.schemas import GenerationMetric
from evaluation.schemas import LatencyStats
from evaluation.schemas import QueryEvaluation

logger = logging.getLogger(__name__)

RetrieveFn = Callable[[str, int], list[str]]

# (query, retrieved_chunk_ids) -> (answer, retrieved_chunk_texts). Kept as a
# plain callable, same philosophy as RetrieveFn - the runner doesn't need to
# know about Answerer/RetrievedChunk/Chunk types, just something that can
# turn ids into an answer plus the context text used to produce it.
AnswerFn = Callable[[str, list[str]], tuple[str, list[str]]]

DEFAULT_K_VALUES = (1, 3, 5, 10)


def current_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return result.stdout.strip() or None


class EvaluationRunner:
    """
    Runs a GoldenDataset against an already-populated retriever and
    produces an EvaluationReport. Ingestion (populating the retriever
    with the dataset's source documents) is the caller's job - the
    runner only needs something shaped like `retrieve_fn(query, top_k)
    -> list[chunk_id]`, so it works the same whether that's a raw
    HybridRetriever, a reranked RAGService pipeline, or a test double.
    Adding a new retrieval metric means adding a function to metrics.py
    and a line in _compute_query_metrics/_aggregate - nothing about this
    class's shape changes.
    """

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        k_values: list[int] | None = None,
        answer_fn: AnswerFn | None = None,
        generation_metrics: list[GenerationMetric] | None = None,
        generation_top_k: int | None = None
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.k_values = k_values or list(DEFAULT_K_VALUES)
        self.answer_fn = answer_fn
        self.generation_metrics = generation_metrics or []
        self.generation_top_k = generation_top_k or min(self.k_values)

    def run(
        self,
        dataset: GoldenDataset,
        metadata: ExperimentMetadata
    ) -> EvaluationReport:
        logger.info(
            "evaluation_started",
            extra={"dataset": dataset.name, "query_count": len(dataset.queries)}
        )

        query_evaluations: list[QueryEvaluation] = []
        latencies: list[float] = []
        retrieved_by_query: list[list[str]] = []
        relevant_by_query: list[set[str]] = []
        max_k = max(self.k_values)

        for query in dataset.queries:
            started_at = time.monotonic()
            retrieved_ids = self.retrieve_fn(query.query, max_k)
            latency_seconds = time.monotonic() - started_at

            relevant_ids = set(query.relevant_chunk_ids)
            query_metrics = self._compute_query_metrics(retrieved_ids, relevant_ids)
            answer, generation_scores = self._compute_generation_metrics(query.query, retrieved_ids)

            latencies.append(latency_seconds)
            retrieved_by_query.append(retrieved_ids)
            relevant_by_query.append(relevant_ids)

            query_evaluations.append(
                QueryEvaluation(
                    query_id=query.id,
                    query=query.query,
                    retrieved_chunk_ids=retrieved_ids,
                    relevant_chunk_ids=query.relevant_chunk_ids,
                    metrics=query_metrics,
                    retrieval_latency_seconds=latency_seconds,
                    category=query.category,
                    difficulty=query.difficulty,
                    answer=answer,
                    generation_metrics=generation_scores
                )
            )

            logger.info(
                "query_evaluated",
                extra={
                    "query_id": query.id,
                    "latency_seconds": round(latency_seconds, 4)
                }
            )

        report = EvaluationReport(
            metadata=metadata,
            k_values=self.k_values,
            aggregate_metrics=self._aggregate(
                query_evaluations,
                retrieved_by_query,
                relevant_by_query
            ),
            query_evaluations=query_evaluations,
            retrieval_latency=self._latency_stats(latencies)
        )

        logger.info(
            "evaluation_completed",
            extra={"dataset": dataset.name, "query_count": len(dataset.queries)}
        )

        return report

    def _compute_query_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str]
    ) -> dict[str, float]:
        query_metrics: dict[str, float] = {}

        for k in self.k_values:
            query_metrics[f"recall@{k}"] = metrics.recall_at_k(retrieved_ids, relevant_ids, k)
            query_metrics[f"precision@{k}"] = metrics.precision_at_k(retrieved_ids, relevant_ids, k)
            query_metrics[f"hit_rate@{k}"] = metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k)
            query_metrics[f"ndcg@{k}"] = metrics.ndcg_at_k(retrieved_ids, relevant_ids, k)

        return query_metrics

    def _compute_generation_metrics(
        self,
        query: str,
        retrieved_ids: list[str]
    ) -> tuple[str | None, dict[str, float]]:
        if self.answer_fn is None or not self.generation_metrics:
            return None, {}

        answer, retrieved_chunk_texts = self.answer_fn(
            query,
            retrieved_ids[:self.generation_top_k]
        )

        return answer, {
            metric.name: metric.score(query, answer, retrieved_chunk_texts)
            for metric in self.generation_metrics
        }

    def _aggregate(
        self,
        query_evaluations: list[QueryEvaluation],
        retrieved_by_query: list[list[str]],
        relevant_by_query: list[set[str]]
    ) -> dict[str, float]:
        aggregate: dict[str, float] = {}

        for k in self.k_values:
            aggregate[f"recall@{k}"] = self._mean(query_evaluations, f"recall@{k}")
            aggregate[f"precision@{k}"] = self._mean(query_evaluations, f"precision@{k}")
            aggregate[f"hit_rate@{k}"] = self._mean(query_evaluations, f"hit_rate@{k}")
            aggregate[f"ndcg@{k}"] = self._mean(query_evaluations, f"ndcg@{k}")

        aggregate["mrr"] = metrics.mean_reciprocal_rank(retrieved_by_query, relevant_by_query)
        aggregate["average_rank"] = metrics.average_rank(retrieved_by_query, relevant_by_query)
        aggregate["average_retrieved_documents"] = metrics.average_retrieved_documents(
            retrieved_by_query
        )

        for metric in self.generation_metrics:
            aggregate[f"generation/{metric.name}"] = self._generation_mean(
                query_evaluations,
                metric.name
            )

        return aggregate

    def _generation_mean(
        self,
        query_evaluations: list[QueryEvaluation],
        metric_name: str
    ) -> float:
        """
        NaN scores (an LLM-judge metric that failed open, see
        evaluation.generation_metrics.LLMJudgeGenerationMetric) are excluded
        from the mean rather than counted as 0 - same "don't penalize what
        we couldn't measure" treatment as average_rank excluding zero-hit
        queries.
        """
        values = [
            qe.generation_metrics[metric_name]
            for qe in query_evaluations
            if metric_name in qe.generation_metrics
            and not math.isnan(qe.generation_metrics[metric_name])
        ]

        if not values:
            return 0.0

        return sum(values) / len(values)

    def _mean(
        self,
        query_evaluations: list[QueryEvaluation],
        metric_name: str
    ) -> float:
        if not query_evaluations:
            return 0.0

        return sum(
            qe.metrics[metric_name] for qe in query_evaluations
        ) / len(query_evaluations)

    def _latency_stats(
        self,
        latencies: list[float]
    ) -> LatencyStats:
        if not latencies:
            return LatencyStats(average_seconds=0.0, median_seconds=0.0, p95_seconds=0.0)

        sorted_latencies = sorted(latencies)

        return LatencyStats(
            average_seconds=sum(latencies) / len(latencies),
            median_seconds=statistics.median(sorted_latencies),
            p95_seconds=self._percentile(sorted_latencies, 0.95)
        )

    def _percentile(
        self,
        sorted_values: list[float],
        percentile: float
    ) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]

        index = min(
            int(round(percentile * (len(sorted_values) - 1))),
            len(sorted_values) - 1
        )
        return sorted_values[index]
