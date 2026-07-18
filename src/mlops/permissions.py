from mlops.schemas import Permission
from mlops.schemas import Role

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.ADMINISTRATOR: frozenset(Permission),
    Role.ML_ENGINEER: frozenset({
        Permission.VIEW,
        Permission.REGISTER_ASSET,
        Permission.PROMOTE_ASSET,
        Permission.EDIT_CONFIGURATION,
        Permission.TOGGLE_FEATURE_FLAG,
        Permission.TRIGGER_DEPLOYMENT,
        Permission.TRIGGER_RETRAINING,
        Permission.TRIGGER_BACKUP,
    }),
    Role.DATA_SCIENTIST: frozenset({
        Permission.VIEW,
        Permission.REGISTER_ASSET,
        Permission.TRIGGER_RETRAINING,
    }),
    Role.REVIEWER: frozenset({
        Permission.VIEW,
        Permission.APPROVE_PROMOTION,
    }),
    Role.READ_ONLY: frozenset({
        Permission.VIEW,
    }),
}


class PermissionDeniedError(PermissionError):
    pass


def has_permission(
    role: Role,
    permission: Permission
) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def permissions_for(
    role: Role
) -> frozenset[Permission]:
    return ROLE_PERMISSIONS.get(role, frozenset())


def require_permission(
    role: Role,
    permission: Permission
) -> None:
    """
    RBAC role-check only - there is no authentication here. Callers are
    responsible for establishing who the actor is (e.g. via their own
    auth/session layer); this only answers "given a role, is this
    action allowed."
    """
    if not has_permission(role, permission):
        raise PermissionDeniedError(
            f"role '{role.value}' lacks permission '{permission.value}'"
        )
