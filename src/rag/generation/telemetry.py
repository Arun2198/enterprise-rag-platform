import logging

from opentelemetry import metrics

logger = logging.getLogger(__name__)

METER_NAME = "enterprise_rag_platform.generation"

_meter = metrics.get_meter(METER_NAME)

generation_requests_total = _meter.create_counter(
    name="generation.requests",
    description="LLM generation calls made"
)
input_tokens_total = _meter.create_counter(
    name="generation.input_tokens",
    description="Prompt/input tokens consumed"
)
output_tokens_total = _meter.create_counter(
    name="generation.output_tokens",
    description="Completion/output tokens generated"
)
estimated_cost_usd_total = _meter.create_counter(
    name="generation.estimated_cost_usd",
    description="Estimated USD cost of LLM generation calls, from a static per-model price table"
)

# USD per 1,000 tokens, (input, output). Deliberately a short, explicit
# table rather than a pricing API integration - "keep implementation
# lightweight" per this project's own cost-observability scope. Prices
# checked against each provider's public pricing page at the time this
# was written; re-verify before trusting this for real budget decisions,
# since LLM API pricing changes without notice and this table doesn't
# auto-update. A model not in this table gets cost=None (see
# record_generation) rather than a silently wrong guess.
MODEL_COST_PER_1K_TOKENS: dict[str, tuple[float, float]] = {
    "anthropic.claude-3-haiku-20240307-v1:0": (0.00025, 0.00125),
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": (0.001, 0.005),
    "gpt-4o-mini": (0.00015, 0.0006),
    "meta/llama-3.1-8b-instruct": (0.0002, 0.0002),
}


def _resolve_model_key(model_id: str) -> str | None:
    """
    Bedrock model_id is sometimes a full inference-profile ARN
    (arn:aws:bedrock:...:inference-profile/global.anthropic.claude-...)
    rather than a plain model id - match on the ARN's trailing segment
    too, so the cost table doesn't need one entry per region/account.
    """
    if model_id in MODEL_COST_PER_1K_TOKENS:
        return model_id

    tail = model_id.rsplit("/", 1)[-1]
    return tail if tail in MODEL_COST_PER_1K_TOKENS else None


def estimate_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int
) -> float | None:
    """
    None means "this model isn't in the price table" - the caller should
    treat that as genuinely unknown cost, not zero cost.
    """
    key = _resolve_model_key(model_id)

    if key is None:
        return None

    input_price, output_price = MODEL_COST_PER_1K_TOKENS[key]
    return (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price


def record_generation(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_seconds: float
) -> None:
    """
    Records one LLM generation call's token usage and estimated cost.
    Called from inside each LLM-backed Answerer right after a successful
    response - never from RAGService or the API layer, since only the
    Answerer actually has the provider's real usage numbers. Never
    raises, same as rag.guardrails.telemetry - a broken exporter or an
    unexpected response shape must not break the answer that was already
    successfully generated.
    """
    try:
        attributes = {"provider": provider, "model": model}
        generation_requests_total.add(1, attributes)

        if input_tokens is None or output_tokens is None:
            # Provider response didn't include usage (e.g. a non-standard
            # OpenAI-compatible endpoint) - record the call happened,
            # skip token/cost metrics rather than recording zeros that
            # would understate real usage.
            logger.info(
                "generation_usage_unavailable",
                extra={"provider": provider, "model": model}
            )
            return

        input_tokens_total.add(input_tokens, attributes)
        output_tokens_total.add(output_tokens, attributes)
        cost = estimate_cost_usd(model, input_tokens, output_tokens)

        if cost is not None:
            estimated_cost_usd_total.add(cost, attributes)

        logger.info(
            "generation_usage_recorded",
            extra={
                "provider": provider,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": cost,
                "latency_seconds": round(latency_seconds, 3)
            }
        )
    except Exception:
        logger.warning("failed to record generation telemetry", exc_info=True)
