import logging
import time

from openai import OpenAI

from rag.generation.prompt import build_grounded_prompt
from rag.retrieval.hybrid_retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

FALLBACK_NO_CONTEXT = "I do not know based on the provided context."
FALLBACK_ERROR = "I couldn't generate an answer at this time."
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenAICompatibleAnswerer:
    """
    Answerer backed by any OpenAI-compatible Chat Completions endpoint
    (OpenAI, Azure OpenAI, GitHub Models, Ollama, OpenRouter, Groq, ...).
    Swapping providers only requires changing base_url/api_key/model_name -
    no code changes.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        timeout: float = 30.0,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        if not retrieved_chunks:
            return FALLBACK_NO_CONTEXT

        prompt = build_grounded_prompt(query, retrieved_chunks)
        started_at = time.monotonic()
        retry_count = 0
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout=self.timeout
                )
            except Exception as ex:
                last_error = ex

                if not self._is_retryable(ex) or attempt == self.max_retries:
                    break

                retry_count += 1
                time.sleep(self.backoff_base_seconds * (2 ** attempt))
                continue

            self._log_success(
                started_at=started_at,
                retry_count=retry_count
            )
            return (response.choices[0].message.content or "").strip()

        self._log_failure(
            started_at=started_at,
            retry_count=retry_count,
            error=last_error
        )
        return FALLBACK_ERROR

    def _is_retryable(
        self,
        error: Exception
    ) -> bool:
        return getattr(error, "status_code", None) in RETRYABLE_STATUS_CODES

    def _log_success(
        self,
        started_at: float,
        retry_count: int
    ) -> None:
        logger.info(
            "llm_answer_succeeded",
            extra={
                "provider": "openai_compatible",
                "model": self.model_name,
                "latency_seconds": round(time.monotonic() - started_at, 3),
                "retry_count": retry_count
            }
        )

    def _log_failure(
        self,
        started_at: float,
        retry_count: int,
        error: Exception | None
    ) -> None:
        logger.warning(
            "llm_answer_failed",
            extra={
                "provider": "openai_compatible",
                "model": self.model_name,
                "latency_seconds": round(time.monotonic() - started_at, 3),
                "retry_count": retry_count,
                "error_type": type(error).__name__ if error is not None else None
            }
        )
