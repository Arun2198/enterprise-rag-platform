from typing import Protocol

from rag.retrieval.hybrid_retrieval import RetrievedChunk


class Answerer(Protocol):

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        ...
