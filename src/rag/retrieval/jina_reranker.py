import logging
import time
from dataclasses import replace

import requests

from rag.retrieval.errors import RerankerProviderError
from rag.retrieval.hybrid_retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker:
    """
    API-based reranker using Jina's rerank endpoint - same interface as
    CrossEncoderReranker (rerank(query, candidates, top_k) -> list[RetrievedChunk])
    so RAGService._retrieve() can use either one interchangeably.

    Live-verified against the real Jina API (see
    scripts/jina_live_verification.py) - correctly ranked a genuinely
    relevant chunk ("water boils at 100 degrees Celsius") at 0.7295 above
    an irrelevant one ("the Eiffel Tower is in Paris") at 0.0284 for a
    boiling-point query. Covered by mocked-HTTP unit tests day-to-day
    since no funded key is available in CI; re-run the verification
    script with a real JINA_API_KEY before trusting a future change to
    this class.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "jina-reranker-v2-base-multilingual",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5
    ) -> None:
        self.provider_name = "jina"
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self._api_key = api_key

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        started_at = time.monotonic()
        body = self._request_with_retry({
            "model": self.model_name,
            "query": query,
            "documents": [candidate.chunk.text for candidate in candidates],
            "top_n": min(top_k, len(candidates))
        })

        reranked = [
            replace(
                candidates[result["index"]],
                score=result["relevance_score"],
                rank=rank
            )
            for rank, result in enumerate(body["results"], start=1)
        ]

        logger.info(
            "reranking_completed",
            extra={
                "provider": self.provider_name,
                "model": self.model_name,
                "candidate_count": len(candidates),
                "reranked_count": len(reranked),
                "latency_seconds": round(time.monotonic() - started_at, 3)
            }
        )
        return reranked

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

            return response.json()

        logger.warning(
            "reranking_request_failed",
            extra={
                "provider": self.provider_name,
                "latency_seconds": round(time.monotonic() - started_at, 3),
                "error_type": type(last_error).__name__ if last_error is not None else None
            }
        )
        raise RerankerProviderError(
            f"jina reranking request failed after {attempt + 1} attempt(s): {last_error}"
        ) from last_error

    def _is_retryable(
        self,
        error: requests.RequestException
    ) -> bool:
        status_code = getattr(error.response, "status_code", None)

        if status_code is not None:
            return status_code in RETRYABLE_STATUS_CODES

        return isinstance(error, (requests.ConnectionError, requests.Timeout))
