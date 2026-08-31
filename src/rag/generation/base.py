from typing import Protocol

from rag.generation.prompt import ConversationTurn
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class Answerer(Protocol):

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None
    ) -> str:
        ...
