import time
from dataclasses import dataclass

from rag.chunking.chunk import Chunk
from rag.embeddings.base import Embedder
from rag.retrieval.trace import CandidateTrace
from rag.retrieval.trace import RetrievalTrace
from rag.vector_store.base import VectorStore
from rag.vector_store.in_memory_store import SearchResult


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    vector_score: float
    keyword_score: float
    score: float
    retrieval_method: str = "dense"
    rank: int = 0


class HybridRetriever:
    """
    True hybrid retrieval: independent dense (vector) and lexical (BM25)
    searches, combined with Reciprocal Rank Fusion - not a linear blend of
    raw scores from two different, incomparable scales (cosine similarity
    and BM25 don't live on the same numeric range, so averaging them
    directly is not a principled fusion). RRF only needs each list's rank
    order, which is why it's the standard first choice for combining
    heterogeneous rankers.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed(query)
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.dense_top_k,
            metadata_filter=metadata_filter
        )
        lexical_results = self.vector_store.search_lexical(
            query_text=query,
            top_k=self.bm25_top_k,
            metadata_filter=metadata_filter
        )

        return self._reciprocal_rank_fusion(dense_results, lexical_results)[:top_k]

    def retrieve_with_trace(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """
        Same retrieval as retrieve(), but also records the raw dense and
        BM25 candidate lists (not just the fused result) plus per-stage
        timing, for RAGService.ask_with_trace()'s debugging endpoint.
        """
        trace = RetrievalTrace(
            query=query,
            embedding_provider=getattr(self.embedder, "provider_name", None),
            embedding_dimensions=getattr(self.embedder, "dimensions", None)
        )

        embed_started = time.monotonic()
        query_embedding = self.embedder.embed(query)
        trace.stage_timings_ms["embedding"] = (time.monotonic() - embed_started) * 1000

        dense_started = time.monotonic()
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.dense_top_k,
            metadata_filter=metadata_filter
        )
        trace.stage_timings_ms["dense_search"] = (time.monotonic() - dense_started) * 1000
        trace.dense_candidates = [
            CandidateTrace(chunk_id=result.chunk.chunk_id, score=result.score, rank=rank)
            for rank, result in enumerate(dense_results, start=1)
        ]

        bm25_started = time.monotonic()
        lexical_results = self.vector_store.search_lexical(
            query_text=query,
            top_k=self.bm25_top_k,
            metadata_filter=metadata_filter
        )
        trace.stage_timings_ms["bm25_search"] = (time.monotonic() - bm25_started) * 1000
        trace.bm25_candidates = [
            CandidateTrace(chunk_id=result.chunk.chunk_id, score=result.score, rank=rank)
            for rank, result in enumerate(lexical_results, start=1)
        ]

        fusion_started = time.monotonic()
        fused = self._reciprocal_rank_fusion(dense_results, lexical_results)
        trace.stage_timings_ms["rrf_fusion"] = (time.monotonic() - fusion_started) * 1000
        trace.fused_candidates = [
            CandidateTrace(chunk_id=item.chunk.chunk_id, score=item.score, rank=item.rank)
            for item in fused
        ]

        return fused[:top_k], trace

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        lexical_results: list[SearchResult]
    ) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        methods: dict[str, set[str]] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunks_by_id[chunk_id] = result.chunk
            vector_scores[chunk_id] = result.score
            methods.setdefault(chunk_id, set()).add("dense")

        for rank, result in enumerate(lexical_results, start=1):
            chunk_id = result.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            chunks_by_id[chunk_id] = result.chunk
            keyword_scores[chunk_id] = result.score
            methods.setdefault(chunk_id, set()).add("bm25")

        ranked_ids = sorted(
            rrf_scores,
            key=lambda chunk_id: rrf_scores[chunk_id],
            reverse=True
        )

        fused = []

        for rank, chunk_id in enumerate(ranked_ids, start=1):
            method_set = methods[chunk_id]
            retrieval_method = "both" if len(method_set) == 2 else next(iter(method_set))
            fused.append(
                RetrievedChunk(
                    chunk=chunks_by_id[chunk_id],
                    vector_score=vector_scores.get(chunk_id, 0.0),
                    keyword_score=keyword_scores.get(chunk_id, 0.0),
                    score=rrf_scores[chunk_id],
                    retrieval_method=retrieval_method,
                    rank=rank
                )
            )

        return fused
