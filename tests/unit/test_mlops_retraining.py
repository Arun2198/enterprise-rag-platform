from mlops.retraining import RETRAINING_TRIGGERS
from mlops.schemas import RetrainingRequest


class FakeRetrainingTrigger:
    """A minimal concrete implementation, proving RetrainingTrigger's shape is usable."""

    def request(self, asset_id, trigger, reason=None) -> RetrainingRequest:
        return RetrainingRequest(
            asset_id=asset_id,
            trigger=trigger,
            requested_at="2026-01-01T00:00:00+00:00",
            reason=reason
        )


def test_expected_triggers_are_documented():

    assert set(RETRAINING_TRIGGERS) == {"scheduled", "drift", "manual"}


def test_fake_trigger_produces_a_retraining_request():

    trigger = FakeRetrainingTrigger()

    request = trigger.request("embedding_model:hashing:1.0", "drift", reason="embedding drift over threshold")

    assert isinstance(request, RetrainingRequest)
    assert request.trigger == "drift"
    assert request.asset_id == "embedding_model:hashing:1.0"
