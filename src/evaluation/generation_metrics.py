import json
import logging
import re
import time

from openai import OpenAI

from rag.embeddings.base import Embedder

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

UNAVAILABLE_SCORE = float("nan")

JUDGE_PROMPT_TEMPLATE = (
    "You are grading a RAG system's answer for overall quality. Given a QUESTION, the "
    "retrieved CONTEXT, and the generated ANSWER, score how good the answer is as a "
    "response to the question, considering both faithfulness to the context and "
    "relevance to the question.\n\n"
    "Respond with ONLY a JSON object, no other text: "
    '{{"quality_score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}\n'
    "1.0 means the answer is fully grounded in the context and directly answers the "
    "question. 0.0 means it is unsupported, irrelevant, or contradicts the context.\n\n"
    "QUESTION:\n{query}\n\n"
    "CONTEXT:\n{context_text}\n\n"
    "ANSWER:\n{answer}\n\n"
    "JSON:"
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    first_norm = sum(a * a for a in first) ** 0.5
    second_norm = sum(b * b for b in second) ** 0.5

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return numerator / (first_norm * second_norm)


class GroundednessMetric:
    """
    How much of the answer's vocabulary actually shows up in the retrieved
    context, blended with embedding cosine similarity when an embedder is
    available - same scoring approach as
    rag.guardrails.hallucination_detector.HallucinationDetector, just
    exposed here as an evaluation-time score instead of a live guardrail
    finding. Kept as a separate small implementation rather than importing
    the guardrail class directly, since evaluation is meant to stay
    standalone from the app/guardrails wiring.
    """
    name = "groundedness"

    def __init__(
        self,
        embedder: Embedder | None = None,
        token_overlap_weight: float = 0.6,
        similarity_weight: float = 0.4
    ) -> None:
        self.embedder = embedder
        self.token_overlap_weight = token_overlap_weight
        self.similarity_weight = similarity_weight

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunk_texts: list[str]
    ) -> float:
        context_text = " ".join(retrieved_chunk_texts)
        answer_terms = _tokens(answer)
        context_terms = _tokens(context_text)

        if not answer_terms or not context_terms:
            return 0.0

        token_overlap = len(answer_terms.intersection(context_terms)) / len(answer_terms)

        if self.embedder is None:
            return token_overlap

        similarity = _cosine_similarity(
            self.embedder.embed(answer),
            self.embedder.embed(context_text)
        )
        blended = (
            self.token_overlap_weight * token_overlap
            + self.similarity_weight * similarity
        )
        return max(0.0, min(blended, 1.0))


class AnswerRelevanceMetric:
    """
    Embedding cosine similarity between the query and the answer - a cheap
    proxy for "does this answer actually address the question", independent
    of whether it's grounded in the context (that's GroundednessMetric's
    job). A confidently wrong answer to a different question scores low
    here even if it happens to be well-grounded in some unrelated chunk.
    """
    name = "answer_relevance"

    def __init__(
        self,
        embedder: Embedder
    ) -> None:
        self.embedder = embedder

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunk_texts: list[str]
    ) -> float:
        if not answer.strip():
            return 0.0

        similarity = _cosine_similarity(
            self.embedder.embed(query),
            self.embedder.embed(answer)
        )
        return max(0.0, min(similarity, 1.0))


class ContextRelevanceMetric:
    """
    Average token overlap between the query and each retrieved chunk -
    measures whether retrieval handed the generator relevant material at
    all, independent of what the generator did with it. Complements the
    golden-dataset id-based recall@k/precision@k from Layer 1 with a
    reference-free signal that still works on queries outside the golden
    dataset.
    """
    name = "context_relevance"

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunk_texts: list[str]
    ) -> float:
        query_terms = _tokens(query)

        if not query_terms or not retrieved_chunk_texts:
            return 0.0

        overlaps = []

        for chunk_text in retrieved_chunk_texts:
            chunk_terms = _tokens(chunk_text)

            if not chunk_terms:
                overlaps.append(0.0)
                continue

            overlaps.append(len(query_terms.intersection(chunk_terms)) / len(query_terms))

        return sum(overlaps) / len(overlaps)


class LLMJudgeGenerationMetric:
    """
    Opt-in composite quality score (faithfulness + relevance in one
    judgment) from an LLM judge, reusing the same OpenAI-compatible client
    pattern as rag.generation.OpenAICompatibleAnswerer and
    rag.guardrails.llm_judge_hallucination_detector. Fails open by
    returning NaN (not 0.0) on any API/parse failure, so a judge outage
    drops that query from the aggregate mean instead of dragging it toward
    "bad" - mirrors how Layer 1's average_rank excludes zero-hit queries
    rather than penalizing them.
    """
    name = "llm_judge_quality"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )

    def score(
        self,
        query: str,
        answer: str,
        retrieved_chunk_texts: list[str]
    ) -> float:
        context_text = " ".join(retrieved_chunk_texts)

        if not answer.strip() or not context_text.strip():
            return UNAVAILABLE_SCORE

        verdict = self._call_judge(query, answer, context_text)

        if verdict is None:
            return UNAVAILABLE_SCORE

        return max(0.0, min(verdict.get("quality_score", 0.0), 1.0))

    def _call_judge(
        self,
        query: str,
        answer: str,
        context_text: str
    ) -> dict | None:
        prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, context_text=context_text, answer=answer)
        started_at = time.monotonic()
        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0,
                    timeout=self.timeout
                )
            except Exception as ex:
                last_error = ex

                if not self._is_retryable(ex) or attempt == self.max_retries:
                    break

                retry_count += 1
                time.sleep(self.backoff_base_seconds * (2 ** attempt))
                continue

            self._log_result("judge_succeeded", started_at, retry_count)
            content = (response.choices[0].message.content or "").strip()
            return self._parse_verdict(content)

        self._log_result("judge_failed", started_at, retry_count, error=last_error)
        return None

    def _parse_verdict(
        self,
        content: str
    ) -> dict | None:
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            pass

        match = re.search(r"\{.*\}", content, re.DOTALL)

        if match is None:
            return None

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _is_retryable(
        self,
        error: Exception
    ) -> bool:
        return getattr(error, "status_code", None) in RETRYABLE_STATUS_CODES

    def _log_result(
        self,
        event: str,
        started_at: float,
        retry_count: int,
        error: Exception | None = None
    ) -> None:
        extra = {
            "metric": self.name,
            "model": self.model_name,
            "latency_seconds": round(time.monotonic() - started_at, 3),
            "retry_count": retry_count
        }

        if error is not None:
            extra["error_type"] = type(error).__name__
            logger.warning(event, extra=extra)
        else:
            logger.info(event, extra=extra)
