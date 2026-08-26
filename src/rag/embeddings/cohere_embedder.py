import logging
import time

import requests

from rag.embeddings.errors import EmbeddingProviderError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://api.cohere.com/v2/embed"


class CohereEmbedder:
    """
    API-based embedder using Cohere's v2 embed endpoint - no local model
    download, real batching (one request for many texts).

    Not live-verified against the real Cohere API in this codebase (that
    needs a real, funded API key this repo doesn't have) - covered by
    mocked-HTTP unit tests only. Verify against a real key before relying
    on it in production.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "embed-english-v3.0",
        dimensions: int = 1024,
        input_type: str = "search_document",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.provider_name = "cohere"
        self.model_name = model_name
        self.dimensions = dimensions
        self.input_type = input_type
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._api_key = api_key

    def embed(
        self,
        text: str
    ) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        if not texts:
            return []

        body = self._request_with_retry({
            "model": self.model_name,
            "texts": texts,
            "input_type": self.input_type,
            "embedding_types": ["float"]
        })
        return body["embeddings"]["float"]

    def _request_with_retry(
        self,
        payload: dict
    ) -> dict:
        started_at = time.monotonic()
        last_error: Exception | None = None
        attempt = 0

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
            except requests.RequestException as ex:
                last_error = ex

                if not self._is_retryable(ex) or attempt == self.max_retries:
                    break

                time.sleep(self.backoff_base_seconds * (2 ** attempt))
                continue

            self._log_success(started_at, attempt, len(payload["texts"]))
            return response.json()

        self._log_failure(started_at, last_error)
        raise EmbeddingProviderError(
            f"cohere embedding request failed after {attempt + 1} attempt(s): {last_error}"
        ) from last_error

    def _is_retryable(
        self,
        error: requests.RequestException
    ) -> bool:
        status_code = getattr(error.response, "status_code", None)

        if status_code is not None:
            return status_code in RETRYABLE_STATUS_CODES

        return isinstance(error, (requests.ConnectionError, requests.Timeout))

    def _log_success(
        self,
        started_at: float,
        retry_count: int,
        batch_size: int
    ) -> None:
        logger.info(
            "embedding_request_succeeded",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "batch_size": batch_size,
                "latency_seconds": round(time.monotonic() - started_at, 3),
                "retry_count": retry_count
            }
        )

    def _log_failure(
        self,
        started_at: float,
        error: Exception | None
    ) -> None:
        logger.warning(
            "embedding_request_failed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "latency_seconds": round(time.monotonic() - started_at, 3),
                "error_type": type(error).__name__ if error is not None else None
            }
        )
