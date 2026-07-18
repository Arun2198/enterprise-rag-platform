import logging
import time

from sentence_transformers import CrossEncoder
from torch import sigmoid

from rag.retrieval.hybrid_retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks hybrid-retrieval candidates by jointly scoring (query, chunk)
    pairs with a cross-encoder. Bi-encoder embeddings encode the query and
    each chunk independently and can miss negation, comparisons, numeric
    or temporal constraints, and word order - a cross-encoder sees both
    texts together and catches those.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        started_at = time.monotonic()
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        # ms-marco-style cross-encoders return raw logits; squash through
        # sigmoid so the score stays a bounded, ranking-preserving [0, 1]
        # relevance signal (RetrievedChunk.score also feeds AskResponse
        # confidence downstream).
        scores = self.model.predict(pairs, activation_fn=sigmoid)

        reranked = sorted(
            (
                self._with_score(candidate, float(score))
                for candidate, score in zip(candidates, scores)
            ),
            key=lambda item: item.score,
            reverse=True
        )[:top_k]

        logger.info(
            "reranking_completed",
            extra={
                "model": self.model_name,
                "candidate_count": len(candidates),
                "reranked_count": len(reranked),
                "latency_seconds": round(time.monotonic() - started_at, 3)
            }
        )

        return reranked

    def _with_score(
        self,
        candidate: RetrievedChunk,
        score: float
    ) -> RetrievedChunk:
        chunk = candidate.chunk.model_copy(
            update={
                "metadata": {
                    **candidate.chunk.metadata,
                    "cross_encoder_score": score
                }
            }
        )
        return RetrievedChunk(
            chunk=chunk,
            vector_score=candidate.vector_score,
            keyword_score=candidate.keyword_score,
            score=score
        )
