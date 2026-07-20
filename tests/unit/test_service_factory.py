from unittest.mock import patch

import pytest

from app.config import Settings
from app.service_factory import ServiceConfigurationError
from app.service_factory import build_platform_manager
from app.service_factory import build_rag_service
from app.services.rag_service import RERANKER_FLAG_NAME
from app.services.rag_service import RAGService
from mlops.manager import PlatformManager


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


@patch("app.service_factory.PresidioPIIGuard")
def test_build_rag_service_wires_presidio_when_enabled(mock_presidio_class):

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        presidio_pii_guard_enabled=True,
        presidio_score_threshold=0.7,
        presidio_entities=("PERSON", "EMAIL_ADDRESS")
    )

    service = build_rag_service(settings)

    assert service.guardrail_manager.guardrails == [mock_presidio_class.return_value]
    mock_presidio_class.assert_called_once_with(
        entities=("PERSON", "EMAIL_ADDRESS"),
        score_threshold=0.7
    )


def test_build_rag_service_skips_presidio_when_disabled():

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        presidio_pii_guard_enabled=False
    )

    service = build_rag_service(settings)

    assert service.guardrail_manager.guardrails == []


@patch("app.service_factory.NLIHallucinationDetector")
def test_build_rag_service_wires_nli_when_enabled(mock_nli_class):

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        nli_hallucination_enabled=True,
        nli_model_name="cross-encoder/custom-nli",
        nli_threshold=0.4
    )

    service = build_rag_service(settings)

    assert service.guardrail_manager.guardrails == [mock_nli_class.return_value]
    mock_nli_class.assert_called_once_with(
        model_name="cross-encoder/custom-nli",
        threshold=0.4
    )


@patch("app.service_factory.LLMJudgeHallucinationDetector")
def test_build_rag_service_wires_llm_judge_when_enabled(mock_judge_class):

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        llm_judge_enabled=True,
        llm_judge_base_url="https://judge.example.com/v1",
        llm_judge_api_key="judge-key",
        llm_judge_model_name="judge-model",
        llm_judge_threshold=0.7
    )

    service = build_rag_service(settings)

    assert service.guardrail_manager.guardrails == [mock_judge_class.return_value]
    mock_judge_class.assert_called_once_with(
        api_key="judge-key",
        base_url="https://judge.example.com/v1",
        model_name="judge-model",
        threshold=0.7
    )


@patch("app.service_factory.LLMJudgeHallucinationDetector")
def test_build_rag_service_llm_judge_falls_back_to_main_llm_settings(mock_judge_class):

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        llm_judge_enabled=True,
        llm_base_url="https://main-llm.example.com/v1",
        llm_api_key="main-key",
        llm_model_name="main-model"
    )

    build_rag_service(settings)

    mock_judge_class.assert_called_once_with(
        api_key="main-key",
        base_url="https://main-llm.example.com/v1",
        model_name="main-model",
        threshold=settings.llm_judge_threshold
    )


def test_build_rag_service_requires_credentials_for_llm_judge():

    settings = Settings(
        pii_guard_enabled=False,
        hallucination_guard_enabled=False,
        llm_judge_enabled=True
    )

    with pytest.raises(ServiceConfigurationError):
        build_rag_service(settings)


def test_build_rag_service_has_no_feature_flags_when_disabled():

    settings = Settings(feature_flags_enabled=False)

    service = build_rag_service(settings)

    assert service.feature_flags is None


def test_build_rag_service_defines_reranker_flag_when_enabled():

    settings = Settings(feature_flags_enabled=True, reranker_rollout_percentage=42.0)

    service = build_rag_service(settings)

    flag = service.feature_flags.get(RERANKER_FLAG_NAME)
    assert flag.enabled is True
    assert flag.rollout_percentage == 42.0


def test_build_platform_manager_returns_none_when_mlops_disabled():

    assert build_platform_manager(Settings(mlops_enabled=False)) is None


def test_build_platform_manager_returns_instance_when_enabled():

    manager = build_platform_manager(Settings(mlops_enabled=True))

    assert isinstance(manager, PlatformManager)


def test_build_platform_manager_registers_backup_job_when_scheduler_enabled():

    manager = build_platform_manager(
        Settings(mlops_enabled=True, scheduler_enabled=True)
    )

    job_ids = [job.job_id for job in manager.scheduler.list_jobs()]
    assert "backup" in job_ids


def test_build_platform_manager_skips_backup_job_when_scheduler_disabled():

    manager = build_platform_manager(
        Settings(mlops_enabled=True, scheduler_enabled=False)
    )

    assert manager.scheduler.list_jobs() == []


def test_build_rag_service_shares_flags_from_platform_manager():

    platform_manager = build_platform_manager(
        Settings(mlops_enabled=True, feature_flags_enabled=True)
    )
    service = build_rag_service(
        Settings(feature_flags_enabled=True),
        platform_manager=platform_manager
    )

    assert service.feature_flags is platform_manager.feature_flags
