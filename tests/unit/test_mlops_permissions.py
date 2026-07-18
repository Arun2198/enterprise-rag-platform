import pytest

from mlops.permissions import PermissionDeniedError
from mlops.permissions import has_permission
from mlops.permissions import permissions_for
from mlops.permissions import require_permission
from mlops.schemas import Permission
from mlops.schemas import Role


def test_administrator_has_every_permission():

    for permission in Permission:
        assert has_permission(Role.ADMINISTRATOR, permission) is True


def test_read_only_has_only_view():

    assert permissions_for(Role.READ_ONLY) == frozenset({Permission.VIEW})


def test_ml_engineer_can_promote_but_not_approve():

    assert has_permission(Role.ML_ENGINEER, Permission.PROMOTE_ASSET) is True
    assert has_permission(Role.ML_ENGINEER, Permission.APPROVE_PROMOTION) is False


def test_reviewer_can_approve_but_not_promote():

    assert has_permission(Role.REVIEWER, Permission.APPROVE_PROMOTION) is True
    assert has_permission(Role.REVIEWER, Permission.PROMOTE_ASSET) is False


def test_data_scientist_can_register_and_retrain_but_not_manage_secrets():

    assert has_permission(Role.DATA_SCIENTIST, Permission.REGISTER_ASSET) is True
    assert has_permission(Role.DATA_SCIENTIST, Permission.TRIGGER_RETRAINING) is True
    assert has_permission(Role.DATA_SCIENTIST, Permission.MANAGE_SECRETS) is False


def test_require_permission_passes_silently_when_allowed():

    require_permission(Role.ADMINISTRATOR, Permission.MANAGE_SECRETS)


def test_require_permission_raises_when_denied():

    with pytest.raises(PermissionDeniedError):
        require_permission(Role.READ_ONLY, Permission.TRIGGER_DEPLOYMENT)
