import pytest

from app.config import Settings
from app.service_factory import ServiceConfigurationError
from app.service_factory import build_rag_service
from app.services.rag_service import RAGService


def test_build_rag_service_defaults_to_local_runtime():

    service = build_rag_service(Settings())

    assert isinstance(service, RAGService)


def test_build_rag_service_rejects_unwired_provider():

    settings = Settings(vector_store_provider="opensearch")

    with pytest.raises(ServiceConfigurationError):
        build_rag_service(settings)
