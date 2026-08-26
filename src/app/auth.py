from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient

from mlops.schemas import Role

DEFAULT_ROLE_CLAIM = "role"


class AuthenticationError(Exception):
    pass


@dataclass(frozen=True)
class AuthenticatedUser:
    subject: str
    role: Role
    claims: dict[str, Any]


class OIDCTokenValidator:
    """
    Generic OIDC/JWT validation - works with any identity provider (AWS
    Cognito, Auth0, Okta, Keycloak, a plain self-hosted OIDC server, ...)
    through plain configuration (issuer/audience/jwks_url), with no
    vendor-specific code anywhere in the validation path. Which identity
    provider to actually run in production is a deployment decision, not
    something this class picks for you.

    Validates: signature (against the issuer's published JWKS, matched by
    the token's "kid" header), issuer, audience, and expiry - all via
    PyJWT's own checks, not reimplemented here. Does not accept
    unsigned/"none"-algorithm tokens or algorithm confusion (RS256 is
    pinned explicitly, never taken from the token itself).
    """

    def __init__(
        self,
        issuer: str,
        audience: str,
        jwks_url: str,
        role_claim: str = DEFAULT_ROLE_CLAIM,
        default_role: Role = Role.READ_ONLY,
        jwks_client: PyJWKClient | None = None
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.role_claim = role_claim
        self.default_role = default_role
        self._jwks_client = jwks_client or PyJWKClient(jwks_url)

    def validate(
        self,
        token: str
    ) -> AuthenticatedUser:
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer
            )
        except jwt.PyJWTError as ex:
            raise AuthenticationError(str(ex)) from ex

        subject = claims.get("sub")

        if not subject:
            raise AuthenticationError("token is missing the required 'sub' claim")

        role = self._resolve_role(claims)
        return AuthenticatedUser(subject=subject, role=role, claims=claims)

    def _resolve_role(
        self,
        claims: dict[str, Any]
    ) -> Role:
        role_value = claims.get(self.role_claim)

        if role_value is None:
            return self.default_role

        try:
            return Role(role_value)
        except ValueError:
            raise AuthenticationError(
                f"token role claim {self.role_claim!r} has an unrecognized value: {role_value!r}"
            )
