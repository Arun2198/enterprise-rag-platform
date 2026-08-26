import json
import logging
import sys
import time

from opentelemetry.sdk.metrics.export import Histogram
from opentelemetry.sdk.metrics.export import MetricExporter
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.metrics.export import MetricsData
from opentelemetry.sdk.metrics.export import Sum

# Separate logger/handler so EMF JSON lines never get mixed into the app's
# normal structured log stream or reformatted by whatever handler main.py
# configures for everything else - CloudWatch Logs only auto-detects a log
# entry as an EMF metric when the line is exactly the raw EMF JSON object,
# nothing prepended.
_emf_logger = logging.getLogger("enterprise_rag_platform.cloudwatch_emf")
_emf_logger.propagate = False
_emf_logger.setLevel(logging.INFO)

if not _emf_logger.handlers:
    # Explicit handler to stdout, independent of whatever the root
    # logger is configured with - ECS's awslogs driver only captures a
    # container's stdout/stderr, and EMF detection needs the raw JSON
    # line with nothing else on it (no timestamp/level prefix a default
    # formatter would add).
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _emf_logger.addHandler(_handler)


class CloudWatchEMFMetricExporter(MetricExporter):
    """
    Exports OpenTelemetry metrics as CloudWatch Embedded Metric Format
    (EMF) log lines, rather than calling PutMetricData directly. EMF is
    the sidecar-free way to get real CloudWatch Metrics out of a plain
    ECS Fargate task: CloudWatch Logs auto-detects the `_aws` EMF
    structure in any log entry sent through the log group this task
    already writes to (the `awslogs` driver in the task definition), and
    creates/updates the named metrics automatically - no ADOT collector,
    no extra IAM permission beyond what logging to CloudWatch already
    needs (`logs:PutLogEvents`, already granted via the execution role).

    Histograms are exported as their mean (sum/count) per collection
    interval, not full bucket fidelity - a deliberate, documented
    simplification (see the guardrail.latency/groundedness_score use
    case this exists for: "is average latency creeping up" matters far
    more here than percentile-accurate histograms, and EMF's own
    high-resolution array format would be real added complexity for
    signal this project doesn't need yet).
    """

    def __init__(
        self,
        namespace: str = "EnterpriseRAGPlatform",
        dimension_keys: tuple[str, ...] = ("stage", "guardrail")
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.dimension_keys = dimension_keys

    def export(
        self,
        metrics_data: MetricsData,
        timeout_millis: float = 10_000,
        **kwargs
    ) -> MetricExportResult:
        try:
            for resource_metrics in metrics_data.resource_metrics:
                for scope_metrics in resource_metrics.scope_metrics:
                    for metric in scope_metrics.metrics:
                        self._emit_metric(metric)

            return MetricExportResult.SUCCESS
        except Exception:
            _emf_logger.warning("cloudwatch_emf_export_failed", exc_info=True)
            return MetricExportResult.FAILURE

    def _emit_metric(
        self,
        metric
    ) -> None:
        if isinstance(metric.data, Sum):
            for sum_point in metric.data.data_points:
                attributes = dict(sum_point.attributes or {})
                self._write_emf(metric.name, sum_point.value, "Count", attributes)
        elif isinstance(metric.data, Histogram):
            for hist_point in metric.data.data_points:
                attributes = dict(hist_point.attributes or {})
                mean = (hist_point.sum / hist_point.count) if hist_point.count else 0.0
                self._write_emf(f"{metric.name}.avg", mean, "None", attributes)
                self._write_emf(f"{metric.name}.count", hist_point.count, "Count", attributes)
        # Gauge/ExponentialHistogram: not currently produced by this
        # project's instruments (see rag/guardrails/telemetry.py and
        # mlops/telemetry.py) - intentionally not handled rather than
        # guessed at without a real instrument to verify against.

    def _write_emf(
        self,
        metric_name: str,
        value: float,
        unit: str,
        attributes: dict
    ) -> None:
        present_dimensions = [key for key in self.dimension_keys if key in attributes]
        record = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": self.namespace,
                        "Dimensions": [present_dimensions] if present_dimensions else [[]],
                        "Metrics": [{"Name": metric_name, "Unit": unit}]
                    }
                ]
            },
            metric_name: value,
            **{key: str(attributes[key]) for key in present_dimensions}
        }
        _emf_logger.info(json.dumps(record))

    def force_flush(
        self,
        timeout_millis: float = 10_000
    ) -> bool:
        return True

    def shutdown(
        self,
        timeout_millis: float = 30_000,
        **kwargs
    ) -> None:
        pass
