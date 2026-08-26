from dataclasses import dataclass
from math import sqrt

from rag.chunking.chunk import Chunk
from rag.retrieval.bm25 import score_bm25
from rag.retrieval.bm25 import tokenize


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

    def search_lexical(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """
        Real BM25 (see rag.retrieval.bm25) over the current record set,
        computed fresh per call - same brute-force-on-every-call philosophy
        as search()'s cosine similarity, and the same real algorithm the
        OpenSearch-backed store's search_lexical() uses, so local/test
        retrieval isn't scored by a different, weaker method than
        production.
        """
        query_terms = tokenize(query_text)
        candidates = [
            (chunk.chunk_id, chunk)
            for chunk, _ in self._records.values()
            if not metadata_filter or self._matches_filter(chunk, metadata_filter)
        ]
        documents = [(chunk_id, tokenize(chunk.text)) for chunk_id, chunk in candidates]
        scores = score_bm25(query_terms, documents)
        chunk_by_id = dict(candidates)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            SearchResult(chunk=chunk_by_id[chunk_id], score=score)
            for chunk_id, score in ranked
        ]

    def get(
        self,
        chunk_id: str
    ) -> Chunk | None:
        record = self._records.get(chunk_id)
        return record[0] if record else None

    def delete(
        self,
        chunk_id: str
    ) -> None:
        self._records.pop(chunk_id, None)

    def delete_by_document(
        self,
        document_id: str
    ) -> int:
        """Removes every chunk for a document. Returns how many chunks
        were removed, so document-lifecycle callers (RAGService.delete_document)
        can report something meaningful rather than a bare None."""
        to_remove = [
            chunk_id
            for chunk_id, (chunk, _) in self._records.items()
            if chunk.document_id == document_id
        ]
        for chunk_id in to_remove:
            del self._records[chunk_id]
        return len(to_remove)

    def update_metadata(
        self,
        chunk_id: str,
        metadata: dict
    ) -> None:
        record = self._records.get(chunk_id)

        if record is None:
            return

        chunk, embedding = record
        self._records[chunk_id] = (chunk.model_copy(update={"metadata": metadata}), embedding)

    def __len__(self) -> int:
        return len(self._records)

    def count(self) -> int:
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
        numerator = sum(a * b for a, b in zip(first, second, strict=True))
        first_norm = sqrt(sum(a * a for a in first))
        second_norm = sqrt(sum(b * b for b in second))

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return numerator / (first_norm * second_norm)
