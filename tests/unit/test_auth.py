import time
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.auth import AuthenticationError
from app.auth import OIDCTokenValidator
from mlops.schemas import Role

ISSUER = "https://auth.example.com/"
AUDIENCE = "enterprise-rag-platform"


@pytest.fixture(scope="module")
def rsa_keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _make_token(private_key, claims_override=None, headers=None):
    now = int(time.time())
    claims = {
        "sub": "user-123",
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "exp": now + 3600,
        "role": "read_only",
    }
    claims.update(claims_override or {})
    return jwt.encode(claims, private_key, algorithm="RS256", headers=headers or {})


def _validator_with_fake_jwks(public_key):
    fake_signing_key = MagicMock()
    fake_signing_key.key = public_key
    fake_jwks_client = MagicMock()
    fake_jwks_client.get_signing_key_from_jwt.return_value = fake_signing_key

    return OIDCTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url="https://auth.example.com/.well-known/jwks.json",
        jwks_client=fake_jwks_client
    )


def test_validates_a_correctly_signed_token(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key)
    validator = _validator_with_fake_jwks(public_key)

    user = validator.validate(token)

    assert user.subject == "user-123"
    assert user.role == Role.READ_ONLY


def test_rejects_a_token_signed_by_a_different_key(rsa_keypair):

    _, public_key = rsa_keypair
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _make_token(other_private_key)
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_rejects_an_expired_token(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"exp": int(time.time()) - 60})
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_rejects_a_token_with_the_wrong_issuer(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"iss": "https://not-the-real-issuer.com/"})
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_rejects_a_token_with_the_wrong_audience(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"aud": "some-other-service"})
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_rejects_a_token_missing_the_subject_claim(rsa_keypair):

    private_key, public_key = rsa_keypair
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 3600}
    token = jwt.encode(claims, private_key, algorithm="RS256")
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError, match="sub"):
        validator.validate(token)


def test_defaults_to_read_only_when_no_role_claim_present(rsa_keypair):

    private_key, public_key = rsa_keypair
    now = int(time.time())
    claims = {"sub": "user-123", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 3600}
    token = jwt.encode(claims, private_key, algorithm="RS256")
    validator = _validator_with_fake_jwks(public_key)

    user = validator.validate(token)

    assert user.role == Role.READ_ONLY


def test_maps_an_administrator_role_claim(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"role": "administrator"})
    validator = _validator_with_fake_jwks(public_key)

    user = validator.validate(token)

    assert user.role == Role.ADMINISTRATOR


def test_rejects_an_unrecognized_role_claim_value(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"role": "super-mega-admin"})
    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(token)


def test_honors_a_custom_role_claim_name(rsa_keypair):

    private_key, public_key = rsa_keypair
    token = _make_token(private_key, claims_override={"custom_role": "ml_engineer", "role": None})
    validator = OIDCTokenValidator(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_url="https://auth.example.com/.well-known/jwks.json",
        role_claim="custom_role",
        jwks_client=_validator_with_fake_jwks(public_key)._jwks_client
    )

    user = validator.validate(token)

    assert user.role == Role.ML_ENGINEER


def test_rejects_an_unsigned_none_algorithm_token(rsa_keypair):

    _, public_key = rsa_keypair
    now = int(time.time())
    claims = {"sub": "user-123", "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 3600}
    # jwt.encode with algorithm=None isn't directly supported by PyJWT for
    # signing "none" tokens safely, so build one by hand to prove the
    # validator can't be tricked into accepting it.
    import base64
    import json

    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "typ": "JWT"}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    none_token = (header + b"." + payload + b".").decode()

    validator = _validator_with_fake_jwks(public_key)

    with pytest.raises(AuthenticationError):
        validator.validate(none_token)
