from unittest.mock import patch

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


@patch("app.service_factory.OpenAICompatibleAnswerer")
def test_build_rag_service_wires_openai_compatible_provider(mock_answerer_class):

    settings = Settings(
        generation_provider="openai_compatible",
        llm_base_url="https://example.com/v1",
        llm_api_key="key"
    )

    service = build_rag_service(settings)

    assert isinstance(service, RAGService)
    assert service.answerer is mock_answerer_class.return_value
    mock_answerer_class.assert_called_once_with(
        api_key="key",
        base_url="https://example.com/v1",
        model_name=settings.llm_model_name,
        timeout=settings.llm_timeout_seconds,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature
    )


def test_build_rag_service_requires_llm_credentials_for_openai_compatible():

    settings = Settings(generation_provider="openai_compatible")

    with pytest.raises(ServiceConfigurationError):
        build_rag_service(settings)
