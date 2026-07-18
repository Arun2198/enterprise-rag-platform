from mlops.telemetry import record_audit_event
from mlops.telemetry import record_operation

from conftest import TELEMETRY_READER

# Shares conftest.TELEMETRY_READER with the guardrails telemetry tests -
# see test_guardrails_telemetry.py for why (OTel allows only one
# MeterProvider per process). Assertions use deltas for the same reason.


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
            total += point.value

    return total


def test_record_operation_increments_counter():

    before = _sum_matching("mlops.operations", operation="register_asset", success=True)

    record_operation("register_asset", success=True)

    after = _sum_matching("mlops.operations", operation="register_asset", success=True)
    assert after == before + 1


def test_record_operation_distinguishes_success_and_failure():

    before_success = _sum_matching("mlops.operations", operation="promote", success=True)
    before_failure = _sum_matching("mlops.operations", operation="promote", success=False)

    record_operation("promote", success=False)

    after_success = _sum_matching("mlops.operations", operation="promote", success=True)
    after_failure = _sum_matching("mlops.operations", operation="promote", success=False)
    assert after_success == before_success
    assert after_failure == before_failure + 1


def test_record_audit_event_increments_counter():

    before = _sum_matching("mlops.audit_events", action="lifecycle_transition")

    record_audit_event("lifecycle_transition")

    after = _sum_matching("mlops.audit_events", action="lifecycle_transition")
    assert after == before + 1
