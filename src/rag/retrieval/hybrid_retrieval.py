import math
import re
from dataclasses import dataclass

from rag.chunking.chunk import Chunk
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.vector_store.in_memory_store import InMemoryVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    vector_score: float
    keyword_score: float
    score: float


class HybridRetriever:

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        embedder: HashingEmbedder,
        vector_weight: float = 0.65,
        keyword_weight: float = 0.35
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=max(top_k * 4, top_k),
            metadata_filter=metadata_filter
        )

        query_terms = self._tokens(query)
        retrieved = []

        for result in vector_results:
            keyword_score = self._keyword_score(query_terms, result.chunk.text)
            fused_score = (
                self.vector_weight * result.score
                + self.keyword_weight * keyword_score
            )
            retrieved.append(
                RetrievedChunk(
                    chunk=result.chunk,
                    vector_score=result.score,
                    keyword_score=keyword_score,
                    score=fused_score
                )
            )

        return sorted(
            retrieved,
            key=lambda item: item.score,
            reverse=True
        )[:top_k]

    def _tokens(
        self,
        text: str
    ) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _keyword_score(
        self,
        query_terms: set[str],
        text: str
    ) -> float:
        if not query_terms:
            return 0.0

        text_terms = self._tokens(text)
        overlap = query_terms.intersection(text_terms)
        return math.log1p(len(overlap)) / math.log1p(len(query_terms))
