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


def test_build_rag_service_restricts_ingest_to_the_configured_directory(tmp_path):

    settings = Settings(ingest_allowed_dir=str(tmp_path))

    service = build_rag_service(settings)

    assert service.ingest_allowed_dir == tmp_path.resolve()


def test_build_rag_service_blocks_a_path_outside_the_default_allowed_dir(tmp_path):

    outside_file = tmp_path / "secret.md"
    outside_file.write_text("# Secret\nNot inside sample_documents.", encoding="utf-8")
    service = build_rag_service(Settings())

    response = service.ingest([str(outside_file)])

    assert response.indexed_documents == 0
    assert "PATH_NOT_ALLOWED" in response.errors[0]


def test_build_rag_service_defaults_to_sentence_transformer_embedder():
    """
    sentence_transformer, not hashing, is the default - the hashing
    embedder is a deterministic fallback for direct RAGService()
    construction (tests, scripts), never what the live app silently falls
    back to.
    """
    from rag.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder

    service = build_rag_service(Settings())

    assert isinstance(service.embedder, SentenceTransformerEmbedder)


def test_build_rag_service_wires_hashing_embedder_when_explicitly_requested():

    from rag.embeddings.hashing_embedder import HashingEmbedder

    service = build_rag_service(Settings(embedding_provider="hashing"))

    assert isinstance(service.embedder, HashingEmbedder)


@patch("app.service_factory.SentenceTransformerEmbedder")
def test_build_rag_service_wires_sentence_transformer_embedder(mock_embedder_class):

    settings = Settings(
        embedding_provider="sentence_transformer",
        embedding_model_name="BAAI/bge-base-en-v1.5"
    )

    service = build_rag_service(settings)

    mock_embedder_class.assert_called_once_with(model_name="BAAI/bge-base-en-v1.5")
    assert service.embedder is mock_embedder_class.return_value


@patch("app.service_factory.SentenceTransformerEmbedder")
def test_build_rag_service_shares_embedder_between_retrieval_and_guardrails(mock_embedder_class):

    settings = Settings(
        embedding_provider="sentence_transformer",
        hallucination_guard_enabled=True
    )

    service = build_rag_service(settings)

    hallucination_guards = [
        g for g in service.guardrail_manager.guardrails if g.name == "hallucination_detector"
    ]
    assert len(hallucination_guards) == 1
    assert hallucination_guards[0].embedder is mock_embedder_class.return_value
    assert hallucination_guards[0].embedder is service.embedder


def test_build_rag_service_rejects_unwired_embedding_provider():

    settings = Settings(embedding_provider="openai")

    with pytest.raises(ServiceConfigurationError):
        build_rag_service(settings)


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


@patch("app.service_factory.BedrockAnswerer")
@patch("app.service_factory.boto3")
def test_build_rag_service_wires_bedrock_provider(mock_boto3, mock_answerer_class):

    settings = Settings(
        generation_provider="bedrock",
        aws_region="us-west-2",
        bedrock_model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )

    service = build_rag_service(settings)

    mock_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")
    assert isinstance(service, RAGService)
    assert service.answerer is mock_answerer_class.return_value
    mock_answerer_class.assert_called_once_with(
        client=mock_boto3.client.return_value,
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature
    )


@patch("app.service_factory.BedrockAnswerer")
@patch("app.service_factory.boto3")
def test_build_rag_service_wires_fallback_answerer_when_configured(mock_boto3, mock_answerer_class):

    settings = Settings(
        generation_provider="bedrock",
        generation_fallback_provider="extractive",
        aws_region="us-west-2",
        bedrock_model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )

    service = build_rag_service(settings)

    from rag.generation.extractive_answerer import ExtractiveAnswerer
    from rag.generation.fallback_answerer import FallbackAnswerer

    assert isinstance(service.answerer, FallbackAnswerer)
    assert service.answerer.primary is mock_answerer_class.return_value
    assert isinstance(service.answerer.fallback, ExtractiveAnswerer)


def test_build_rag_service_has_no_fallback_by_default():

    settings = Settings(generation_provider="extractive")

    service = build_rag_service(settings)

    from rag.generation.fallback_answerer import FallbackAnswerer

    assert not isinstance(service.answerer, FallbackAnswerer)


@patch("app.service_factory.OpenAICompatibleAnswerer")
def test_build_rag_service_wires_openai_compatible_as_fallback(mock_answerer_class):

    settings = Settings(
        generation_provider="extractive",
        generation_fallback_provider="openai_compatible",
        llm_base_url="https://models.github.ai/inference",
        llm_api_key="key"
    )

    service = build_rag_service(settings)

    from rag.generation.fallback_answerer import FallbackAnswerer

    assert isinstance(service.answerer, FallbackAnswerer)
    assert service.answerer.fallback is mock_answerer_class.return_value


def test_build_rag_service_requires_llm_credentials_for_openai_compatible_fallback():

    settings = Settings(
        generation_provider="extractive",
        generation_fallback_provider="openai_compatible"
    )

    with pytest.raises(ServiceConfigurationError):
        build_rag_service(settings)


def test_build_rag_service_rejects_unwired_fallback_provider():

    settings = Settings(
        generation_provider="extractive",
        generation_fallback_provider="not_a_real_provider"
    )

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
