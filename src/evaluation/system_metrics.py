from evaluation.schemas import EvaluationReport

CHARS_PER_TOKEN_ESTIMATE = 4.0


class DefaultSystemMetricsCollector:
    """
    Layer 3: system-level metrics computed from an already-finished
    EvaluationReport, plus two optional real measurements a caller can
    supply from wrapping the actual run (run_eval.py does this with
    tracemalloc + a wall clock - see its --system-metrics flag).

    Deliberately does NOT report a guardrail-trigger rate: EvaluationRunner
    never executes guardrails (it only calls retrieve_fn/answer_fn), so
    there is no real trigger data in an EvaluationReport to summarize. A
    caller that wires a GuardrailManager into their own answer_fn could
    compute that themselves and merge it into this dict; fabricating a
    metric that's always 0 here would be actively misleading.

    Token/cost estimates are a documented approximation - they use
    query/answer text length only (~4 chars/token, a common rule of thumb),
    because the retrieved context text itself isn't retained on
    QueryEvaluation once metrics are computed. That means the real
    prompt (which is dominated by retrieved context, not the query) is
    undercounted - treat estimated_cost_usd as a lower bound, not a bill.
    """
    name = "default_system_metrics"

    def __init__(
        self,
        cost_per_1k_completion_tokens: float = 0.0
    ) -> None:
        self.cost_per_1k_completion_tokens = cost_per_1k_completion_tokens

    def collect(
        self,
        report: EvaluationReport,
        run_duration_seconds: float | None = None,
        peak_memory_mb: float | None = None
    ) -> dict[str, float]:
        query_count = len(report.query_evaluations)
        metrics: dict[str, float] = {"queries_evaluated": float(query_count)}

        total_retrieval_seconds = sum(
            qe.retrieval_latency_seconds for qe in report.query_evaluations
        )
        metrics["retrieval_throughput_qps"] = (
            query_count / total_retrieval_seconds if total_retrieval_seconds > 0 else 0.0
        )

        answered = [qe for qe in report.query_evaluations if qe.answer]

        if answered:
            estimated_completion_tokens = [
                len(qe.answer) / CHARS_PER_TOKEN_ESTIMATE for qe in answered
            ]
            estimated_query_tokens = [
                len(qe.query) / CHARS_PER_TOKEN_ESTIMATE for qe in answered
            ]
            total_completion_tokens = sum(estimated_completion_tokens)
            metrics["estimated_completion_tokens_total"] = total_completion_tokens
            metrics["estimated_completion_tokens_avg"] = total_completion_tokens / len(answered)
            metrics["estimated_query_tokens_avg"] = sum(estimated_query_tokens) / len(answered)
            metrics["estimated_completion_cost_usd"] = (
                total_completion_tokens / 1000 * self.cost_per_1k_completion_tokens
            )

        if run_duration_seconds is not None:
            metrics["run_duration_seconds"] = run_duration_seconds
            metrics["overall_throughput_qps"] = (
                query_count / run_duration_seconds if run_duration_seconds > 0 else 0.0
            )

        if peak_memory_mb is not None:
            metrics["peak_memory_mb"] = peak_memory_mb

        return metrics
