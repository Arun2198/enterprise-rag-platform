from typing import Protocol

from rag.chunking.chunk import Chunk
from rag.vector_store.in_memory_store import SearchResult


class VectorStore(Protocol):

    def add(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> None:
        ...

    def add_many(
        self,
        records: list[tuple[Chunk, list[float]]]
    ) -> None:
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        ...
