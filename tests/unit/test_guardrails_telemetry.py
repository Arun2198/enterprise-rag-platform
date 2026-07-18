import pytest

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.telemetry import record_action
from rag.guardrails.telemetry import record_finding

from conftest import TELEMETRY_READER

# OTel only allows a MeterProvider to be set once per process, so this
# module shares conftest.TELEMETRY_READER with every other telemetry
# test file rather than setting up its own; tests assert on deltas
# rather than absolute values or "metric X is absent" (once any test in
# the whole suite emits a metric stream, its name persists for the rest
# of the session).


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


def _sum_matching(metric_name, **attribute_filters):
    total = 0.0

    for point in _metric_points(metric_name):
        if all(point.attributes.get(key) == value for key, value in attribute_filters.items()):
            total += getattr(point, "value", None) if hasattr(point, "value") else point.sum

    return total


def test_pii_finding_records_detection_and_latency():

    before = _sum_matching("guardrail.pii_detections", guardrail="pii_guard")
    finding = GuardrailFinding(
        guardrail_name="pii_guard",
        triggered=True,
        severity=Severity.WARNING,
        action=Action.REDACT,
        message="redacted"
    )

    record_finding(finding, GuardrailStage.OUTPUT, latency_seconds=0.01)

    after = _sum_matching("guardrail.pii_detections", guardrail="pii_guard")
    assert after == before + 1
    assert len(_metric_points("guardrail.latency")) >= 1
    assert len(_metric_points("guardrail.runs")) >= 1


def test_clean_pii_finding_does_not_increment_detection_count():

    before = _sum_matching("guardrail.pii_detections", guardrail="pii_guard")
    finding = GuardrailFinding(
        guardrail_name="pii_guard",
        triggered=False,
        severity=Severity.INFO,
        action=Action.ALLOW,
        message="clean"
    )

    record_finding(finding, GuardrailStage.OUTPUT, latency_seconds=0.01)

    after = _sum_matching("guardrail.pii_detections", guardrail="pii_guard")
    assert after == before


def test_hallucination_finding_records_flag_and_groundedness():

    flags_before = _sum_matching(
        "guardrail.hallucination_flags",
        guardrail="hallucination_detector"
    )
    groundedness_sum_before = _sum_matching(
        "guardrail.groundedness_score",
        guardrail="hallucination_detector"
    )
    finding = GuardrailFinding(
        guardrail_name="hallucination_detector",
        triggered=True,
        severity=Severity.WARNING,
        action=Action.WARN,
        message="ungrounded",
        metadata={"groundedness_score": 0.2}
    )

    record_finding(finding, GuardrailStage.OUTPUT, latency_seconds=0.02)

    flags_after = _sum_matching(
        "guardrail.hallucination_flags",
        guardrail="hallucination_detector"
    )
    groundedness_sum_after = _sum_matching(
        "guardrail.groundedness_score",
        guardrail="hallucination_detector"
    )
    assert flags_after == flags_before + 1
    # histogram .sum is cumulative for the process lifetime, not per-call -
    # assert the delta this call contributed, not an absolute value
    assert groundedness_sum_after - groundedness_sum_before == pytest.approx(0.2)


def test_llm_judge_finding_is_categorized_as_hallucination_metric():

    points_before = len(_metric_points("guardrail.groundedness_score"))
    finding = GuardrailFinding(
        guardrail_name="llm_judge_hallucination_detector",
        triggered=False,
        severity=Severity.INFO,
        action=Action.ALLOW,
        message="grounded",
        metadata={"groundedness_score": 0.9}
    )

    record_finding(finding, GuardrailStage.OUTPUT, latency_seconds=0.5)

    judge_points = [
        point
        for point in _metric_points("guardrail.groundedness_score")
        if point.attributes.get("guardrail") == "llm_judge_hallucination_detector"
    ]
    assert len(judge_points) >= 1
    assert len(_metric_points("guardrail.groundedness_score")) > points_before


def test_non_hallucination_guardrail_does_not_record_groundedness():

    before = len(_metric_points("guardrail.groundedness_score"))
    finding = GuardrailFinding(
        guardrail_name="secret_leakage_guard",
        triggered=True,
        severity=Severity.CRITICAL,
        action=Action.BLOCK,
        message="secret detected",
        metadata={"detected_entities": ["AWS_ACCESS_KEY"]}
    )

    record_finding(finding, GuardrailStage.OUTPUT, latency_seconds=0.01)

    after = len(_metric_points("guardrail.groundedness_score"))
    assert after == before


def test_record_action_block_increments_blocked_counter():

    before = _sum_matching("guardrail.blocked_responses", stage="output")

    record_action(Action.BLOCK, GuardrailStage.OUTPUT)

    after = _sum_matching("guardrail.blocked_responses", stage="output")
    assert after == before + 1


def test_record_action_allow_does_not_increment_blocked_counter():

    before = _sum_matching("guardrail.blocked_responses", stage="output")

    record_action(Action.ALLOW, GuardrailStage.OUTPUT)

    after = _sum_matching("guardrail.blocked_responses", stage="output")
    assert after == before
