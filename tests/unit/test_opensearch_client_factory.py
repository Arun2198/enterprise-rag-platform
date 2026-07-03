import pytest

from app.config import Settings
from rag.vector_store.opensearch_client_factory import OpenSearchConfigurationError
from rag.vector_store.opensearch_client_factory import build_opensearch_client


def test_opensearch_client_factory_requires_host():

    with pytest.raises(OpenSearchConfigurationError):
        build_opensearch_client(Settings(vector_store_provider="opensearch"))
