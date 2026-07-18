import os
from typing import Protocol


class SecretNotFoundError(KeyError):
    pass


class SecretValue:
    """
    Wraps a secret so it never accidentally ends up in a log line,
    exception message, or repr(). Call .reveal() explicitly to get the
    raw string - that should be the one place a secret value ever
    touches application code.
    """
    __slots__ = ("_value",)

    def __init__(
        self,
        value: str
    ) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretValue(***redacted***)"

    def __str__(self) -> str:
        return "***redacted***"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretValue):
            return self._value == other._value

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


class SecretsProvider(Protocol):
    """
    Same shape for every backend - local .env, Azure Key Vault, AWS
    Secrets Manager, GCP Secret Manager. Only LocalEnvSecretsProvider is
    implemented; the cloud backends are extension points a deployment
    wires in by constructing a client that satisfies this Protocol - no
    caller code changes when swapping one in.
    """

    def get(self, name: str) -> SecretValue:
        ...

    def exists(self, name: str) -> bool:
        ...


class LocalEnvSecretsProvider:
    """
    Reads secrets from process environment variables, matching how
    app/config.py already reads everything else - the local-dev
    default. `prefix` optionally namespaces lookups, e.g.
    prefix="MLOPS_" means get("API_KEY") reads the MLOPS_API_KEY env var.
    """

    def __init__(
        self,
        prefix: str = ""
    ) -> None:
        self.prefix = prefix

    def get(
        self,
        name: str
    ) -> SecretValue:
        env_name = f"{self.prefix}{name}"
        value = os.getenv(env_name)

        if value is None:
            raise SecretNotFoundError(env_name)

        return SecretValue(value)

    def exists(
        self,
        name: str
    ) -> bool:
        return f"{self.prefix}{name}" in os.environ
