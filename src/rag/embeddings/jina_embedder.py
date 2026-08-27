import logging
import time

import requests

from rag.embeddings.errors import EmbeddingProviderError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://api.jina.ai/v1/embeddings"


class JinaEmbedder:
    """
    API-based embedder using Jina AI's embeddings endpoint - no local model
    download, so the AWS deployment doesn't need to fetch/hold a model in
    the container. Every embed call is a signed HTTPS request; batching is
    real (one request for many texts), not simulated by looping embed().

    Live-verified against the real Jina API (see
    scripts/jina_live_verification.py) - a real batch embed_batch() call
    returned two genuine 1024-dim vectors with real token usage reported
    back by the API. Covered by mocked-HTTP unit tests day-to-day since
    no funded key is available in CI; re-run the verification script
    with a real JINA_API_KEY before trusting a future change to this
    class, the same way the OpenSearch client was verified for real.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "jina-embeddings-v3",
        dimensions: int = 1024,
        task: str = "retrieval.passage",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.provider_name = "jina"
        self.model_name = model_name
        self.dimensions = dimensions
        self.task = task
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
            "task": self.task,
            "dimensions": self.dimensions,
            "input": texts
        })
        ordered = sorted(body["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in ordered]

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

            self._log_success(started_at, attempt, len(payload["input"]))
            return response.json()

        self._log_failure(started_at, last_error)
        raise EmbeddingProviderError(
            f"jina embedding request failed after {attempt + 1} attempt(s): {last_error}"
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
