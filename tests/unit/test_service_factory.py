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


def test_build_rag_service_disables_reranking_when_reranker_disabled():

    settings = Settings(reranker_enabled=False)

    service = build_rag_service(settings)

    assert service.reranker is None


@patch("app.service_factory.CrossEncoderReranker")
def test_build_rag_service_wires_reranker_when_enabled(mock_reranker_class):

    settings = Settings(
        reranker_enabled=True,
        reranker_model_name="cross-encoder/custom-model",
        reranker_candidate_multiplier=6
    )

    service = build_rag_service(settings)

    assert service.reranker is mock_reranker_class.return_value
    assert service.candidate_multiplier == 6
    mock_reranker_class.assert_called_once_with(
        model_name="cross-encoder/custom-model"
    )


def test_build_rag_service_disables_guardrails_when_guardrails_disabled():

    settings = Settings(guardrails_enabled=False)

    service = build_rag_service(settings)

    assert service.guardrail_manager.guardrails == []


def test_build_rag_service_wires_guardrails_when_enabled():

    settings = Settings(
        guardrails_enabled=True,
        pii_guard_enabled=True,
        hallucination_guard_enabled=False,
        groundedness_threshold=0.75
    )

    service = build_rag_service(settings)

    guardrail_names = [g.name for g in service.guardrail_manager.guardrails]

    assert guardrail_names == ["pii_guard"]


def test_build_rag_service_wires_hallucination_guard_with_configured_threshold():

    settings = Settings(
        guardrails_enabled=True,
        pii_guard_enabled=False,
        hallucination_guard_enabled=True,
        groundedness_threshold=0.75
    )

    service = build_rag_service(settings)

    hallucination_guards = [
        g for g in service.guardrail_manager.guardrails
        if g.name == "hallucination_detector"
    ]

    assert len(hallucination_guards) == 1
    assert hallucination_guards[0].threshold == 0.75
