import json

from conftest import TELEMETRY_READER

from app.observability import CloudWatchEMFMetricExporter
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.telemetry import record_finding

# Same shared-MeterProvider constraint as test_guardrails_telemetry.py -
# this exports whatever real metrics exist in the process by the time the
# test runs, via the same TELEMETRY_READER every other telemetry test
# shares, rather than standing up a second MeterProvider (OTel only
# allows one per process).


def test_export_writes_valid_emf_json_for_a_counter(caplog):

    record_finding(
        GuardrailFinding(
            guardrail_name="pii_guard",
            triggered=True,
            severity=Severity.WARNING,
            action=Action.REDACT,
            message="test"
        ),
        stage=GuardrailStage.OUTPUT,
        latency_seconds=0.01
    )

    metrics_data = TELEMETRY_READER.get_metrics_data()
    exporter = CloudWatchEMFMetricExporter()

    with caplog.at_level("INFO", logger="enterprise_rag_platform.cloudwatch_emf"):
        result = exporter.export(metrics_data)

    from opentelemetry.sdk.metrics.export import MetricExportResult
    assert result == MetricExportResult.SUCCESS

    emf_lines = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "enterprise_rag_platform.cloudwatch_emf"
    ]
    assert emf_lines

    runs_lines = [line for line in emf_lines if "guardrail.runs" in line]
    assert runs_lines

    emf_envelope = runs_lines[0]["_aws"]
    assert emf_envelope["CloudWatchMetrics"][0]["Namespace"] == "EnterpriseRAGPlatform"
    assert {"Name": "guardrail.runs", "Unit": "Count"} in \
        emf_envelope["CloudWatchMetrics"][0]["Metrics"]
    assert "Timestamp" in emf_envelope


def test_export_writes_histogram_as_mean_and_count(caplog):

    record_finding(
        GuardrailFinding(
            guardrail_name="hallucination_detector",
            triggered=False,
            severity=Severity.INFO,
            action=Action.ALLOW,
            message="test",
            metadata={"groundedness_score": 0.85}
        ),
        stage=GuardrailStage.OUTPUT,
        latency_seconds=0.02
    )

    metrics_data = TELEMETRY_READER.get_metrics_data()
    exporter = CloudWatchEMFMetricExporter()

    with caplog.at_level("INFO", logger="enterprise_rag_platform.cloudwatch_emf"):
        exporter.export(metrics_data)

    emf_lines = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "enterprise_rag_platform.cloudwatch_emf"
    ]

    assert any("guardrail.groundedness_score.avg" in line for line in emf_lines)
    assert any("guardrail.groundedness_score.count" in line for line in emf_lines)


def test_force_flush_and_shutdown_do_not_raise():

    exporter = CloudWatchEMFMetricExporter()

    assert exporter.force_flush() is True
    exporter.shutdown()
