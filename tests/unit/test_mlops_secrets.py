import pytest

from mlops.secrets import LocalEnvSecretsProvider
from mlops.secrets import SecretNotFoundError
from mlops.secrets import SecretValue


def test_get_reads_from_environment(monkeypatch):

    monkeypatch.setenv("MY_API_KEY", "sk-super-secret")
    provider = LocalEnvSecretsProvider()

    secret = provider.get("MY_API_KEY")

    assert secret.reveal() == "sk-super-secret"


def test_get_missing_secret_raises(monkeypatch):

    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    provider = LocalEnvSecretsProvider()

    with pytest.raises(SecretNotFoundError):
        provider.get("DOES_NOT_EXIST")


def test_exists_reflects_environment(monkeypatch):

    monkeypatch.setenv("MY_API_KEY", "value")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    provider = LocalEnvSecretsProvider()

    assert provider.exists("MY_API_KEY") is True
    assert provider.exists("MISSING_KEY") is False


def test_prefix_namespaces_lookups(monkeypatch):

    monkeypatch.setenv("MLOPS_API_KEY", "namespaced-value")
    provider = LocalEnvSecretsProvider(prefix="MLOPS_")

    secret = provider.get("API_KEY")

    assert secret.reveal() == "namespaced-value"


def test_secret_value_never_leaks_in_repr_or_str():

    secret = SecretValue("sk-super-secret")

    assert "sk-super-secret" not in repr(secret)
    assert "sk-super-secret" not in str(secret)
    assert secret.reveal() == "sk-super-secret"


def test_secret_value_equality_compares_underlying_value():

    assert SecretValue("same") == SecretValue("same")
    assert SecretValue("a") != SecretValue("b")
