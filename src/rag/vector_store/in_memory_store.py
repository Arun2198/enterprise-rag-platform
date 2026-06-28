from dataclasses import dataclass
from math import sqrt

from rag.chunking.chunk import Chunk


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class InMemoryVectorStore:

    def __init__(self) -> None:
        self._records: dict[str, tuple[Chunk, list[float]]] = {}

    def add(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> None:
        self._records[chunk.chunk_id] = (chunk, embedding)

    def add_many(
        self,
        records: list[tuple[Chunk, list[float]]]
    ) -> None:
        for chunk, embedding in records:
            self.add(chunk, embedding)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        results = []

        for chunk, embedding in self._records.values():
            if metadata_filter and not self._matches_filter(chunk, metadata_filter):
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=self._cosine_similarity(query_embedding, embedding)
                )
            )

        return sorted(
            results,
            key=lambda result: result.score,
            reverse=True
        )[:top_k]

    def __len__(self) -> int:
        return len(self._records)

    def _matches_filter(
        self,
        chunk: Chunk,
        metadata_filter: dict[str, str]
    ) -> bool:
        return all(
            str(chunk.metadata.get(key)) == str(value)
            for key, value in metadata_filter.items()
        )

    def _cosine_similarity(
        self,
        first: list[float],
        second: list[float]
    ) -> float:
        numerator = sum(a * b for a, b in zip(first, second))
        first_norm = sqrt(sum(a * a for a in first))
        second_norm = sqrt(sum(b * b for b in second))

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return numerator / (first_norm * second_norm)
