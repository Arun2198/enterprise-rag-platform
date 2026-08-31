import logging

from rag.generation.base import Answerer
from rag.generation.prompt import ConversationTurn
from rag.retrieval.hybrid_retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class FallbackAnswerer:
    """
    Wraps two Answerer instances: tries primary first, and on any
    exception - a Bedrock throttling/access/billing error, a network
    failure, whatever - falls back to the secondary instead of surfacing
    a 500 to the caller. Generic composition over the Answerer Protocol;
    doesn't know or care which concrete providers it's wrapping, so any
    two Answerer implementations can be paired this way.
    """

    def __init__(
        self,
        primary: Answerer,
        fallback: Answerer
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None
    ) -> str:
        try:
            return self.primary.answer(query, retrieved_chunks, history=history)
        except Exception as ex:
            logger.warning(
                "primary_answerer_failed_falling_back",
                extra={
                    "primary": type(self.primary).__name__,
                    "fallback": type(self.fallback).__name__,
                    "error_type": type(ex).__name__
                }
            )
            return self.fallback.answer(query, retrieved_chunks, history=history)
