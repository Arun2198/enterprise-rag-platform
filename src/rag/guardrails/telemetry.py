import logging

from opentelemetry import metrics

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage

logger = logging.getLogger(__name__)

METER_NAME = "enterprise_rag_platform.guardrails"

PII_GUARDRAIL_NAMES = frozenset({"pii_guard", "presidio_pii_guard"})
HALLUCINATION_GUARDRAIL_NAMES = frozenset({
    "hallucination_detector",
    "nli_hallucination_detector",
    "llm_judge_hallucination_detector",
})

_meter = metrics.get_meter(METER_NAME)

guardrail_runs_total = _meter.create_counter(
    name="guardrail.runs",
    description="Guardrail checks executed"
)
guardrail_latency_seconds = _meter.create_histogram(
    name="guardrail.latency",
    unit="s",
    description="Latency of a single guardrail check"
)
pii_detections_total = _meter.create_counter(
    name="guardrail.pii_detections",
    description="PII detections across all PII guardrails"
)
hallucination_flags_total = _meter.create_counter(
    name="guardrail.hallucination_flags",
    description="Answers flagged as likely hallucinations"
)
groundedness_score = _meter.create_histogram(
    name="guardrail.groundedness_score",
    description="Groundedness score reported by hallucination detectors"
)
blocked_responses_total = _meter.create_counter(
    name="guardrail.blocked_responses",
    description="Responses blocked by the guardrails pipeline"
)


def record_finding(
    finding: GuardrailFinding,
    stage: GuardrailStage,
    latency_seconds: float
) -> None:
    """
    Records one guardrail check as OpenTelemetry metrics. Works whether or
    not the host application has configured a real MeterProvider - with
    none configured, these are cheap no-ops (the OTel API's default). A
    console or Prometheus exporter can be wired up by the host app without
    any change here. Never raises - a broken exporter must not break the
    guardrail pipeline it's observing.
    """
    try:
        attributes = {"guardrail": finding.guardrail_name, "stage": stage.value}
        guardrail_latency_seconds.record(latency_seconds, attributes)
        guardrail_runs_total.add(
            1,
            {**attributes, "triggered": finding.triggered}
        )

        if finding.triggered and finding.guardrail_name in PII_GUARDRAIL_NAMES:
            pii_detections_total.add(1, attributes)

        if finding.guardrail_name in HALLUCINATION_GUARDRAIL_NAMES:
            if finding.triggered:
                hallucination_flags_total.add(1, attributes)

            score = finding.metadata.get("groundedness_score")

            if score is not None:
                groundedness_score.record(score, attributes)
    except Exception:
        logger.warning("failed to record guardrail metrics", exc_info=True)


def record_action(
    action: Action,
    stage: GuardrailStage
) -> None:
    try:
        if action == Action.BLOCK:
            blocked_responses_total.add(1, {"stage": stage.value})
    except Exception:
        logger.warning("failed to record guardrail action metric", exc_info=True)
