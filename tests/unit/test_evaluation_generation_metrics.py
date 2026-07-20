import math
from unittest.mock import MagicMock
from unittest.mock import patch

from evaluation.generation_metrics import AnswerRelevanceMetric
from evaluation.generation_metrics import ContextRelevanceMetric
from evaluation.generation_metrics import GroundednessMetric
from evaluation.generation_metrics import LLMJudgeGenerationMetric
from rag.embeddings.hashing_embedder import HashingEmbedder


def test_groundedness_scores_high_when_answer_reuses_context_vocabulary():

    metric = GroundednessMetric()

    score = metric.score(
        query="what color is the sky",
        answer="the sky is blue",
        retrieved_chunk_texts=["the sky is blue during the day"]
    )

    assert score == 1.0


def test_groundedness_scores_zero_for_unrelated_answer():

    metric = GroundednessMetric()

    score = metric.score(
        query="what color is the sky",
        answer="bananas are yellow fruit",
        retrieved_chunk_texts=["the sky is blue during the day"]
    )

    assert score == 0.0


def test_groundedness_blends_embedding_similarity_when_embedder_given():

    metric = GroundednessMetric(embedder=HashingEmbedder())

    score = metric.score(
        query="q",
        answer="the sky is blue",
        retrieved_chunk_texts=["the sky is blue during the day"]
    )

    assert 0.0 < score <= 1.0


def test_groundedness_handles_empty_answer_or_context():

    metric = GroundednessMetric()

    assert metric.score("q", "", ["some context"]) == 0.0
    assert metric.score("q", "an answer", []) == 0.0


def test_answer_relevance_scores_identical_text_as_one():

    metric = AnswerRelevanceMetric(embedder=HashingEmbedder())

    score = metric.score(
        query="what is the capital of france",
        answer="what is the capital of france",
        retrieved_chunk_texts=[]
    )

    assert score == 1.0


def test_answer_relevance_handles_empty_answer():

    metric = AnswerRelevanceMetric(embedder=HashingEmbedder())

    assert metric.score("q", "", []) == 0.0


def test_context_relevance_scores_full_overlap():

    metric = ContextRelevanceMetric()

    score = metric.score(
        query="ai risk management framework",
        answer="irrelevant",
        retrieved_chunk_texts=["ai risk management framework overview"]
    )

    assert score == 1.0


def test_context_relevance_handles_no_retrieved_chunks():

    metric = ContextRelevanceMetric()

    assert metric.score("q", "a", []) == 0.0


@patch("evaluation.generation_metrics.OpenAI")
def test_llm_judge_parses_valid_json_response(mock_openai_cls):

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(message=MagicMock(content='{"quality_score": 0.85, "reasoning": "good"}'))
        ]
    )
    metric = LLMJudgeGenerationMetric(api_key="k", base_url="http://x", model_name="m")

    score = metric.score(
        query="q",
        answer="a",
        retrieved_chunk_texts=["context"]
    )

    assert score == 0.85


@patch("evaluation.generation_metrics.OpenAI")
def test_llm_judge_fails_open_with_nan_on_api_error(mock_openai_cls):

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("boom")
    metric = LLMJudgeGenerationMetric(api_key="k", base_url="http://x", model_name="m", max_retries=0)

    score = metric.score(
        query="q",
        answer="a",
        retrieved_chunk_texts=["context"]
    )

    assert math.isnan(score)


def test_llm_judge_returns_nan_without_calling_client_when_no_context():

    metric = LLMJudgeGenerationMetric(api_key="k", base_url="http://x", model_name="m")

    score = metric.score(query="q", answer="a", retrieved_chunk_texts=[])

    assert math.isnan(score)
