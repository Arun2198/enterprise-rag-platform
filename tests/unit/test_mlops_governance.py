import pytest

from mlops.governance import GovernanceLog
from mlops.governance import PolicyViolationError
from mlops.schemas import ApprovalRecord
from mlops.schemas import LifecycleStage
from mlops.schemas import LifecycleTransition


def test_record_appends_an_audit_event():

    log = GovernanceLog()

    event = log.record(actor="alice", action="register_asset", resource="asset-1")

    assert event.actor == "alice"
    assert log.history() == [event]


def test_record_transition_captures_stage_change():

    log = GovernanceLog()
    transition = LifecycleTransition(
        asset_id="asset-1",
        from_stage=LifecycleStage.DEVELOPMENT,
        to_stage=LifecycleStage.VALIDATION,
        timestamp="2026-01-01T00:00:00+00:00",
        actor="alice"
    )

    event = log.record_transition(transition)

    assert event.action == "lifecycle_transition"
    assert event.details["from_stage"] == "development"
    assert event.details["to_stage"] == "validation"


def test_record_approval_captures_approver():

    log = GovernanceLog()
    approval = ApprovalRecord(
        asset_id="asset-1",
        to_stage=LifecycleStage.STAGING,
        approved_by="bob",
        timestamp="2026-01-01T00:00:00+00:00"
    )

    event = log.record_approval(approval)

    assert event.actor == "bob"
    assert event.details["to_stage"] == "staging"


def test_link_lineage_and_query():

    log = GovernanceLog()

    log.link_lineage("asset-1", "chunking-config", 2)

    assert log.lineage("asset-1") == {"chunking-config": 2}


def test_lineage_for_unknown_asset_is_empty():

    log = GovernanceLog()

    assert log.lineage("does-not-exist") == {}


def test_history_can_filter_by_resource():

    log = GovernanceLog()
    log.record(actor=None, action="a", resource="asset-1")
    log.record(actor=None, action="b", resource="asset-2")

    assert len(log.history(resource="asset-1")) == 1
    assert len(log.history()) == 2


def test_check_policy_passes_and_records_event():

    log = GovernanceLog()

    log.check_policy("no_pii_in_prod", condition=True, message="clean")

    event = log.history()[-1]
    assert event.details["passed"] is True


def test_check_policy_failure_raises_and_still_records_event():

    log = GovernanceLog()

    with pytest.raises(PolicyViolationError):
        log.check_policy("no_pii_in_prod", condition=False, message="PII detected")

    event = log.history()[-1]
    assert event.details["passed"] is False
