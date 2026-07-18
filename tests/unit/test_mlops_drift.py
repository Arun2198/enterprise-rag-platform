from mlops.drift import DRIFT_TYPES
from mlops.schemas import DriftReport


class FakeDriftDetector:
    """A minimal concrete implementation, proving DriftDetector's shape is usable."""

    drift_type = "embedding"

    def detect(self, *args, **kwargs) -> DriftReport:
        return DriftReport(
            drift_type=self.drift_type,
            detected=False,
            score=0.02,
            threshold=0.1,
            timestamp="2026-01-01T00:00:00+00:00"
        )


def test_expected_drift_types_are_documented():

    assert set(DRIFT_TYPES) == {
        "data", "embedding", "retrieval", "prompt", "model", "user_query"
    }


def test_fake_detector_satisfies_the_interface_shape():

    detector = FakeDriftDetector()

    report = detector.detect()

    assert isinstance(report, DriftReport)
    assert report.drift_type == "embedding"
    assert report.detected is False
