import pytest

from mlops.feature_flags import FeatureFlagManager
from mlops.feature_flags import FlagNotFoundError


def test_define_and_get():

    manager = FeatureFlagManager()

    flag = manager.define("enable_reranker", enabled=True, description="Toggle the reranker")

    assert flag.enabled is True
    assert manager.get("enable_reranker") == flag


def test_get_missing_flag_raises():

    manager = FeatureFlagManager()

    with pytest.raises(FlagNotFoundError):
        manager.get("does-not-exist")


def test_set_enabled_toggles_flag():

    manager = FeatureFlagManager()
    manager.define("enable_guardrails", enabled=False)

    updated = manager.set_enabled("enable_guardrails", True)

    assert updated.enabled is True


def test_disabled_flag_is_never_enabled_for_anyone():

    manager = FeatureFlagManager()
    manager.define("enable_reranker", enabled=False, rollout_percentage=100.0)

    assert manager.is_enabled_for("enable_reranker", "user-1") is False


def test_full_rollout_is_enabled_for_everyone():

    manager = FeatureFlagManager()
    manager.define("enable_reranker", enabled=True, rollout_percentage=100.0)

    assert manager.is_enabled_for("enable_reranker", "user-1") is True
    assert manager.is_enabled_for("enable_reranker", "user-2") is True


def test_zero_rollout_is_disabled_for_everyone():

    manager = FeatureFlagManager()
    manager.define("canary", enabled=True, rollout_percentage=0.0)

    assert manager.is_enabled_for("canary", "user-1") is False


def test_canary_bucketing_is_deterministic_per_subject():

    manager = FeatureFlagManager()
    manager.define("canary", enabled=True, rollout_percentage=50.0)

    first = manager.is_enabled_for("canary", "user-42")
    second = manager.is_enabled_for("canary", "user-42")

    assert first == second


def test_invalid_rollout_percentage_raises():

    manager = FeatureFlagManager()

    with pytest.raises(ValueError):
        manager.define("canary", rollout_percentage=150.0)


def test_shadow_flag_can_be_toggled():

    manager = FeatureFlagManager()
    manager.define("shadow_llm_provider", enabled=True, shadow=True)

    assert manager.get("shadow_llm_provider").shadow is True

    updated = manager.set_shadow("shadow_llm_provider", False)

    assert updated.shadow is False


def test_export_import_state_round_trips():

    manager = FeatureFlagManager()
    manager.define("enable_reranker", enabled=True, rollout_percentage=75.0)

    state = manager.export_state()
    restored = FeatureFlagManager()
    restored.import_state(state)

    assert restored.get("enable_reranker") == manager.get("enable_reranker")
