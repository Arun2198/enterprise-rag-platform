from conftest import TELEMETRY_READER

from rag.generation.telemetry import estimate_cost_usd
from rag.generation.telemetry import record_generation


def _metric_points(metric_name):
    data = TELEMETRY_READER.get_metrics_data()

    if data is None:
        return []

    points = []

    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == metric_name:
                    points.extend(metric.data.data_points)

    return points


def test_estimate_cost_for_a_known_model():

    cost = estimate_cost_usd("gpt-4o-mini", input_tokens=1000, output_tokens=1000)

    assert cost == 0.00015 + 0.0006


def test_estimate_cost_resolves_bedrock_inference_profile_arn_to_the_plain_model_key():

    arn = (
        "arn:aws:bedrock:us-east-1:849279003696:inference-profile/"
        "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    cost = estimate_cost_usd(arn, input_tokens=1000, output_tokens=1000)

    assert cost == 0.001 + 0.005


def test_estimate_cost_is_none_for_an_unknown_model():

    cost = estimate_cost_usd("some-brand-new-model-not-in-the-table", input_tokens=1000, output_tokens=1000)

    assert cost is None


def test_record_generation_records_requests_tokens_and_cost():

    record_generation(
        provider="openai_compatible",
        model="gpt-4o-mini",
        input_tokens=500,
        output_tokens=100,
        latency_seconds=1.2
    )

    requests = [
        p for p in _metric_points("generation.requests")
        if p.attributes.get("model") == "gpt-4o-mini"
    ]
    tokens_in = [
        p for p in _metric_points("generation.input_tokens")
        if p.attributes.get("model") == "gpt-4o-mini"
    ]
    cost = [
        p for p in _metric_points("generation.estimated_cost_usd")
        if p.attributes.get("model") == "gpt-4o-mini"
    ]

    assert requests
    assert tokens_in
    assert tokens_in[-1].value >= 500
    assert cost


def test_record_generation_skips_token_metrics_when_usage_is_unavailable():

    before = len(_metric_points("generation.input_tokens"))

    record_generation(
        provider="openai_compatible",
        model="some-endpoint-without-usage",
        input_tokens=None,
        output_tokens=None,
        latency_seconds=0.5
    )

    after = len(_metric_points("generation.input_tokens"))

    # a brand new attribute combination would add a new data point; since
    # no tokens were recorded for this model, no new point should appear
    assert after == before


def test_record_generation_never_raises_on_unexpected_input():
    record_generation(
        provider="bedrock",
        model="whatever",
        input_tokens="not-a-number",  # type: ignore[arg-type]
        output_tokens=10,
        latency_seconds=0.1
    )
