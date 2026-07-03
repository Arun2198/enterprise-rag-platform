from typing import Any

from app.config import Settings


class OpenSearchConfigurationError(ValueError):
    pass


def build_opensearch_client(
    settings: Settings
) -> Any:
    if not settings.opensearch_host:
        raise OpenSearchConfigurationError(
            "OPENSEARCH_HOST is required when VECTOR_STORE_PROVIDER=opensearch."
        )

    try:
        from opensearchpy import OpenSearch
    except ImportError as ex:
        raise OpenSearchConfigurationError(
            "opensearch-py is required for VECTOR_STORE_PROVIDER=opensearch."
        ) from ex

    http_auth = None

    if settings.opensearch_username and settings.opensearch_password:
        http_auth = (
            settings.opensearch_username,
            settings.opensearch_password
        )

    return OpenSearch(
        hosts=[settings.opensearch_host],
        http_auth=http_auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        timeout=settings.opensearch_timeout_seconds,
    )
