import pytest

from mlops.configuration import ConfigurationManager
from mlops.configuration import ConfigValidationError
from mlops.configuration import NoActiveProfileError
from mlops.configuration import NoRollbackTargetError
from mlops.configuration import ProfileNotFoundError


def test_save_profile_creates_version_one():

    manager = ConfigurationManager()

    profile = manager.save_profile("dev", {"reranker_enabled": True})

    assert profile.version == 1


def test_saving_same_profile_name_again_creates_new_version():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    second = manager.save_profile("dev", {"chunk_size": 512})

    assert second.version == 2
    assert manager.history("dev")[0].values == {"chunk_size": 900}


def test_get_without_active_profile_returns_default():

    manager = ConfigurationManager()

    assert manager.get("chunk_size", default=42) == 42


def test_activate_and_get_reads_from_active_profile():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    manager.activate("dev")

    assert manager.get("chunk_size") == 900


def test_override_takes_priority_over_active_profile():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    manager.activate("dev")
    manager.set_override("chunk_size", 256)

    assert manager.get("chunk_size") == 256

    manager.clear_override("chunk_size")

    assert manager.get("chunk_size") == 900


def test_rollback_activates_previous_version():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    manager.save_profile("dev", {"chunk_size": 512})
    manager.activate("dev")

    rolled_back = manager.rollback()

    assert rolled_back.version == 1
    assert manager.get("chunk_size") == 900


def test_rollback_without_earlier_version_raises():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    manager.activate("dev")

    with pytest.raises(NoRollbackTargetError):
        manager.rollback()


def test_rollback_without_active_profile_raises():

    manager = ConfigurationManager()

    with pytest.raises(NoActiveProfileError):
        manager.rollback()


def test_activate_missing_profile_raises():

    manager = ConfigurationManager()

    with pytest.raises(ProfileNotFoundError):
        manager.activate("does-not-exist")


def test_validation_rejects_invalid_value():

    def positive_int(value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("must be a positive int")

    manager = ConfigurationManager(validators={"chunk_size": positive_int})

    with pytest.raises(ConfigValidationError):
        manager.save_profile("dev", {"chunk_size": -1})


def test_validation_allows_valid_value():

    def positive_int(value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("must be a positive int")

    manager = ConfigurationManager(validators={"chunk_size": positive_int})

    profile = manager.save_profile("dev", {"chunk_size": 900})

    assert profile.values["chunk_size"] == 900


def test_export_import_state_round_trips():

    manager = ConfigurationManager()
    manager.save_profile("dev", {"chunk_size": 900})
    manager.activate("dev")
    manager.set_override("reranker_enabled", False)

    state = manager.export_state()
    restored = ConfigurationManager()
    restored.import_state(state)

    assert restored.get("chunk_size") == 900
    assert restored.get("reranker_enabled") is False
