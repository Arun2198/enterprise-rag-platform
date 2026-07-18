import logging

from opentelemetry import metrics

logger = logging.getLogger(__name__)

METER_NAME = "enterprise_rag_platform.mlops"

_meter = metrics.get_meter(METER_NAME)

operations_total = _meter.create_counter(
    name="mlops.operations",
    description="PlatformManager operations executed"
)
audit_events_total = _meter.create_counter(
    name="mlops.audit_events",
    description="Governance audit events recorded"
)


def record_operation(
    operation: str,
    success: bool
) -> None:
    """
    Same pattern as rag.guardrails.telemetry: works with no MeterProvider
    configured (cheap no-op), and a host app can attach a real exporter
    at startup with zero changes here. Never raises.
    """
    try:
        operations_total.add(1, {"operation": operation, "success": success})
    except Exception:
        logger.warning("failed to record mlops operation metric", exc_info=True)


def record_audit_event(
    action: str
) -> None:
    try:
        audit_events_total.add(1, {"action": action})
    except Exception:
        logger.warning("failed to record mlops audit event metric", exc_info=True)
