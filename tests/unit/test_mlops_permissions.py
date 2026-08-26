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


def test_read_only_can_view_and_query_but_nothing_else():

    assert permissions_for(Role.READ_ONLY) == frozenset({Permission.VIEW, Permission.QUERY})


def test_read_only_cannot_upload_or_delete_documents():

    assert has_permission(Role.READ_ONLY, Permission.UPLOAD_DOCUMENT) is False
    assert has_permission(Role.READ_ONLY, Permission.DELETE_DOCUMENT) is False


def test_data_scientist_can_upload_but_not_delete_documents():

    assert has_permission(Role.DATA_SCIENTIST, Permission.UPLOAD_DOCUMENT) is True
    assert has_permission(Role.DATA_SCIENTIST, Permission.DELETE_DOCUMENT) is False


def test_ml_engineer_can_upload_and_delete_documents():

    assert has_permission(Role.ML_ENGINEER, Permission.UPLOAD_DOCUMENT) is True
    assert has_permission(Role.ML_ENGINEER, Permission.DELETE_DOCUMENT) is True


def test_every_role_can_query():

    for role in Role:
        assert has_permission(role, Permission.QUERY) is True


def test_debug_query_gated_from_read_only_and_reviewer():

    assert has_permission(Role.READ_ONLY, Permission.DEBUG_QUERY) is False
    assert has_permission(Role.REVIEWER, Permission.DEBUG_QUERY) is False


def test_debug_query_allowed_for_ml_engineer_and_data_scientist():

    assert has_permission(Role.ML_ENGINEER, Permission.DEBUG_QUERY) is True
    assert has_permission(Role.DATA_SCIENTIST, Permission.DEBUG_QUERY) is True


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
