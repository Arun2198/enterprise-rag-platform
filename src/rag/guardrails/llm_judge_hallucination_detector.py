import json
import logging
import re
import time

from openai import OpenAI

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

JUDGE_PROMPT_TEMPLATE = (
    "You are a strict fact-checking judge. Given CONTEXT and an ANSWER, decide how "
    "well the ANSWER is supported by the CONTEXT alone.\n\n"
    "Respond with ONLY a JSON object, no other text: "
    '{{"groundedness_score": <float 0.0-1.0>, "reasoning": "<one short sentence>"}}\n'
    "1.0 means every claim in the answer is directly supported by the context. "
    "0.0 means the answer is unsupported or contradicts the context.\n\n"
    "CONTEXT:\n{context_text}\n\n"
    "ANSWER:\n{answer}\n\n"
    "JSON:"
)


class LLMJudgeHallucinationDetector:
    """
    Phase 2 example guardrail (output stage): asks an LLM to judge whether
    the generated answer is grounded in the retrieved context, instead of
    a heuristic score. Reuses the same OpenAI-compatible client pattern as
    OpenAICompatibleAnswerer - any OpenAI-compatible endpoint works by
    changing base_url/api_key/model_name, no code changes. Fails open
    (does not trigger, does not block) if the judge call or response
    parsing fails, since an infrastructure hiccup on the judge shouldn't
    itself take down otherwise-fine answers.
    """
    name = "llm_judge_hallucination_detector"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        threshold: float = 0.60,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        answer = context.answer or ""
        context_text = " ".join(
            item.chunk.text for item in context.retrieved_chunks
        )

        if not answer.strip() or not context_text.strip():
            return self._unavailable_finding("no answer or context to judge")

        verdict = self._call_judge(answer, context_text)

        if verdict is None:
            return self._unavailable_finding("judge call failed or response was unparseable")

        score = max(0.0, min(verdict.get("groundedness_score", 0.0), 1.0))
        reasoning = verdict.get("reasoning", "")
        likely_hallucination = score < self.threshold

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=likely_hallucination,
            severity=Severity.WARNING if likely_hallucination else Severity.INFO,
            action=Action.WARN if likely_hallucination else Action.ALLOW,
            message=(
                f"judge groundedness {score:.2f} below threshold {self.threshold:.2f}: {reasoning}"
                if likely_hallucination else
                f"judge groundedness {score:.2f} meets threshold {self.threshold:.2f}"
            ),
            metadata={
                "groundedness_score": round(score, 4),
                "likely_hallucination": likely_hallucination,
                "judge_available": True
            }
        )

    def _unavailable_finding(
        self,
        reason: str
    ) -> GuardrailFinding:
        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=False,
            severity=Severity.INFO,
            action=Action.ALLOW,
            message=f"judge unavailable: {reason}",
            metadata={"judge_available": False}
        )

    def _call_judge(
        self,
        answer: str,
        context_text: str
    ) -> dict | None:
        prompt = JUDGE_PROMPT_TEMPLATE.format(context_text=context_text, answer=answer)
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
            "guardrail": self.name,
            "model": self.model_name,
            "latency_seconds": round(time.monotonic() - started_at, 3),
            "retry_count": retry_count
        }

        if error is not None:
            extra["error_type"] = type(error).__name__
            logger.warning(event, extra=extra)
        else:
            logger.info(event, extra=extra)
