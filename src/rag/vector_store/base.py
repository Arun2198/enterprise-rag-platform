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

    def search_lexical(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        ...

    def delete(
        self,
        chunk_id: str
    ) -> None:
        ...

    def delete_by_document(
        self,
        document_id: str
    ) -> int:
        ...

    def update_metadata(
        self,
        chunk_id: str,
        metadata: dict
    ) -> None:
        ...

    def count(self) -> int:
        ...
