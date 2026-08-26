import time
import uuid
from datetime import datetime
from datetime import timezone
from pathlib import Path

from app.schemas import AskResponse
from app.schemas import IngestResponse
from app.schemas import Source
from ingestion.contracts.document import Document
from ingestion.ingestion_pipeline import IngestionPipeline
from mlops.feature_flags import FeatureFlagManager
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.base import Embedder
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.generation.base import Answerer
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.guardrails.base import Action
from rag.guardrails.manager import GuardrailManager
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.trace import CandidateTrace
from rag.retrieval.trace import RetrievalTrace
from rag.vector_store.base import VectorStore
from rag.vector_store.in_memory_store import InMemoryVectorStore

RERANKER_FLAG_NAME = "cross_encoder_reranker"
ABSTENTION_MESSAGE = (
    "I don't have enough supporting evidence in the indexed documents to "
    "answer this confidently, so I'm not going to guess."
)


class RAGService:

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline | None = None,
        chunker: RecursiveChunker | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        answerer: Answerer | None = None,
        reranker: CrossEncoderReranker | None = None,
        candidate_multiplier: int = 4,
        feature_flags: FeatureFlagManager | None = None,
        guardrail_manager: GuardrailManager | None = None,
        guardrails_enabled: bool = True,
        pii_guard_enabled: bool = True,
        hallucination_guard_enabled: bool = True,
        groundedness_threshold: float = 0.60,
        retrieval_relevance_guard_enabled: bool = False,
        retrieval_relevance_threshold: float | None = None,
        ingest_allowed_dir: str | None = None,
        dense_top_k: int = 20,
        bm25_top_k: int = 20,
        rrf_k: int = 60,
        abstention_enabled: bool = True
    ) -> None:
        self.abstention_enabled = abstention_enabled
        self.ingest_allowed_dir = (
            Path(ingest_allowed_dir).resolve() if ingest_allowed_dir is not None else None
        )
        self.ingestion_pipeline = ingestion_pipeline or IngestionPipeline()
        self.chunker = chunker or RecursiveChunker()
        self.embedder = embedder or HashingEmbedder()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.answerer = answerer or ExtractiveAnswerer()
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.feature_flags = feature_flags
        self.guardrail_manager = guardrail_manager or (
            GuardrailManager.default(
                embedder=self.embedder,
                pii_enabled=pii_guard_enabled,
                hallucination_enabled=hallucination_guard_enabled,
                groundedness_threshold=groundedness_threshold,
                retrieval_relevance_enabled=retrieval_relevance_guard_enabled,
                retrieval_relevance_threshold=retrieval_relevance_threshold
            )
            if guardrails_enabled else
            GuardrailManager(guardrails=[])
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k
        )

    def ingest(
        self,
        file_paths: list[str]
    ) -> IngestResponse:
        indexed_documents = 0
        indexed_chunks = 0
        errors = []

        for file_path in file_paths:
            if not self._is_path_allowed(file_path):
                errors.append(
                    f"{file_path}: PATH_NOT_ALLOWED file path is outside the allowed "
                    "ingestion directory"
                )
                continue

            document_result = self.ingestion_pipeline.ingest_file(file_path)

            if not document_result.success or document_result.data is None:
                errors.append(self._format_error(file_path, document_result.error))
                continue

            chunk_count = self.index_document(document_result.data)

            if chunk_count is None:
                errors.append(f"{file_path}: CHUNKING_FAILED could not chunk document")
                continue

            indexed_documents += 1
            indexed_chunks += chunk_count

        return IngestResponse(
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            errors=errors
        )

    def index_document(
        self,
        document: Document
    ) -> int | None:
        """
        Chunk + embed + index a document that's already been parsed -
        shared by the synchronous local-file ingest() path above and the
        async S3/SQS ingestion worker, which parses via
        IngestionPipeline.ingest_from_s3() itself and only needs this half
        of the pipeline. Returns the chunk count indexed, or None if
        chunking failed.
        """
        chunk_result = self.chunker.chunk(document)

        if not chunk_result.success or chunk_result.data is None:
            return None

        indexed_at = datetime.now(timezone.utc)
        chunks = [
            chunk.model_copy(update={
                "embedding_provider": getattr(self.embedder, "provider_name", None),
                "embedding_model": getattr(self.embedder, "model_name", None),
                "embedding_version": str(self.embedder.dimensions),
                "indexed_at": indexed_at
            })
            for chunk in chunk_result.data
        ]
        embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
        records = list(zip(chunks, embeddings))
        self.vector_store.add_many(records)
        return len(records)

    def delete_document(
        self,
        document_id: str
    ) -> int:
        """
        Removes every indexed chunk for a document - the full document
        lifecycle (upload/update/delete/reindex) needs a real delete path,
        not just the vector-store-level primitive. Returns how many
        chunks were removed (0 if the document_id had none indexed).
        """
        return self.vector_store.delete_by_document(document_id)

    def reindex_document(
        self,
        file_path: str
    ) -> IngestResponse:
        """
        Delete-then-reingest for a document that changed - not a partial
        update, a full replace, since chunk boundaries/count can shift
        with any content change and stale chunks from the old version
        must not survive alongside the new ones.
        """
        document_result = self.ingestion_pipeline.ingest_file(file_path)

        if not document_result.success or document_result.data is None:
            return IngestResponse(
                indexed_documents=0,
                indexed_chunks=0,
                errors=[self._format_error(file_path, document_result.error)]
            )

        self.delete_document(document_result.data.document_id)
        chunk_count = self.index_document(document_result.data)

        if chunk_count is None:
            return IngestResponse(
                indexed_documents=0,
                indexed_chunks=0,
                errors=[f"{file_path}: CHUNKING_FAILED could not chunk document"]
            )

        return IngestResponse(indexed_documents=1, indexed_chunks=chunk_count, errors=[])

    def ask(
        self,
        query: str,
        top_k: int = 5,
        client_id: str | None = None,
        access_groups: list[str] | None = None
    ) -> AskResponse:
        input_result = self.guardrail_manager.run_input(query)

        if input_result.action == Action.BLOCK:
            return AskResponse(
                answer=input_result.text,
                sources=[],
                confidence=0.0,
                guardrail_flags=input_result.flags
            )

        query = input_result.text
        retrieved = self._retrieve(
            query=query,
            top_k=top_k,
            client_id=client_id,
            access_groups=access_groups
        )
        answer = self.answerer.answer(
            query=query,
            retrieved_chunks=retrieved
        )

        output_result = self.guardrail_manager.run_output(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved
        )

        if output_result.action == Action.BLOCK:
            return AskResponse(
                answer=output_result.text,
                sources=[],
                confidence=0.0,
                guardrail_flags=output_result.flags
            )

        sources = [
            Source(
                document_id=item.chunk.document_id,
                document_version=item.chunk.document_version,
                chunk_id=item.chunk.chunk_id,
                section=item.chunk.parent_section,
                source=item.chunk.source,
                score=item.score,
                retrieval_method=item.retrieval_method,
                rank=item.rank,
                text=item.chunk.text
            )
            for item in retrieved
        ]

        groundedness = output_result.flags.get("groundedness")
        confidence = self._compute_confidence(groundedness, retrieved)
        answer_text = output_result.text

        if self.abstention_enabled and self._should_abstain(output_result.flags):
            # HallucinationDetector's own default action is WARN, not
            # BLOCK (a deliberate choice - see manager default: never
            # auto-block on a heuristic score alone) - the guardrail flag
            # stays a warning, but the user-facing answer text still
            # shouldn't be a likely-fabricated claim presented as fact.
            # Sources are kept so the low-groundedness evidence stays
            # auditable even though it wasn't trusted enough to answer
            # from.
            answer_text = ABSTENTION_MESSAGE

        return AskResponse(
            answer=answer_text,
            sources=sources,
            groundedness=groundedness,
            confidence=confidence,
            guardrail_flags=output_result.flags
        )

    def ask_with_trace(
        self,
        query: str,
        top_k: int = 5,
        client_id: str | None = None,
        access_groups: list[str] | None = None
    ) -> tuple[AskResponse, RetrievalTrace]:
        """
        Same behavior as ask(), plus a full per-stage RetrievalTrace
        (embedding/dense/BM25/RRF/rerank/generation/groundedness/guardrail
        detail and latency) - kept as a separate method rather than a flag
        on ask() so the normal request path never pays for trace
        bookkeeping it doesn't use. Gated at the API layer, not here - this
        method has no opinion on who's allowed to call it.
        """
        total_started = time.monotonic()
        input_result = self.guardrail_manager.run_input(query)

        if input_result.action == Action.BLOCK:
            trace = RetrievalTrace(query=query)
            trace.guardrail_findings = input_result.flags.get("details", [])
            trace.stage_timings_ms["total"] = (time.monotonic() - total_started) * 1000
            return AskResponse(
                answer=input_result.text,
                sources=[],
                confidence=0.0,
                guardrail_flags=input_result.flags
            ), trace

        query = input_result.text
        retrieved, trace = self._retrieve_with_trace(
            query=query,
            top_k=top_k,
            client_id=client_id,
            access_groups=access_groups
        )

        generation_started = time.monotonic()
        answer = self.answerer.answer(
            query=query,
            retrieved_chunks=retrieved
        )
        trace.stage_timings_ms["generation"] = (time.monotonic() - generation_started) * 1000
        trace.generation_provider = type(self.answerer).__name__
        trace.final_chunk_ids = [item.chunk.chunk_id for item in retrieved]

        guardrail_started = time.monotonic()
        output_result = self.guardrail_manager.run_output(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved
        )
        trace.stage_timings_ms["output_guardrails"] = (time.monotonic() - guardrail_started) * 1000
        trace.guardrail_findings = output_result.flags.get("details", [])
        trace.groundedness = output_result.flags.get("groundedness")
        trace.stage_timings_ms["total"] = (time.monotonic() - total_started) * 1000

        if output_result.action == Action.BLOCK:
            return AskResponse(
                answer=output_result.text,
                sources=[],
                confidence=0.0,
                guardrail_flags=output_result.flags
            ), trace

        sources = [
            Source(
                document_id=item.chunk.document_id,
                document_version=item.chunk.document_version,
                chunk_id=item.chunk.chunk_id,
                section=item.chunk.parent_section,
                source=item.chunk.source,
                score=item.score,
                retrieval_method=item.retrieval_method,
                rank=item.rank,
                text=item.chunk.text
            )
            for item in retrieved
        ]

        groundedness = output_result.flags.get("groundedness")
        confidence = self._compute_confidence(groundedness, retrieved)
        answer_text = output_result.text

        if self.abstention_enabled and output_result.flags.get("hallucination") is True:
            answer_text = ABSTENTION_MESSAGE

        return AskResponse(
            answer=answer_text,
            sources=sources,
            groundedness=groundedness,
            confidence=confidence,
            guardrail_flags=output_result.flags
        ), trace

    def _retrieve_with_trace(
        self,
        query: str,
        top_k: int,
        client_id: str | None = None,
        access_groups: list[str] | None = None
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        if self.reranker is None or not self._reranker_enabled_for(client_id):
            candidates, trace = self.retriever.retrieve_with_trace(
                query=query,
                top_k=top_k
            )
            return self._filter_by_access(candidates, access_groups)[:top_k], trace

        candidates, trace = self.retriever.retrieve_with_trace(
            query=query,
            top_k=top_k * self.candidate_multiplier
        )
        authorized = self._filter_by_access(candidates, access_groups)

        rerank_started = time.monotonic()
        reranked = self.reranker.rerank(
            query=query,
            candidates=authorized,
            top_k=top_k
        )
        trace.stage_timings_ms["rerank"] = (time.monotonic() - rerank_started) * 1000
        trace.reranker_used = True
        trace.reranked_candidates = [
            CandidateTrace(chunk_id=item.chunk.chunk_id, score=item.score, rank=item.rank)
            for item in reranked
        ]
        return reranked, trace

    def _should_abstain(
        self,
        guardrail_flags: dict
    ) -> bool:
        """
        Two independent, complementary signals can trigger abstention:
        `hallucination` (does the answer match its own retrieved evidence)
        and `low_retrieval_relevance` (was that evidence actually relevant
        to the query in the first place - see RetrievalRelevanceGuard's
        docstring for why groundedness alone can't catch a confidently
        wrong answer built from confidently irrelevant retrieval). Either
        one firing is enough - they catch different failure modes, not the
        same one twice.
        """
        return (
            guardrail_flags.get("hallucination") is True
            or guardrail_flags.get("low_retrieval_relevance") is True
        )

    def _compute_confidence(
        self,
        groundedness: float | None,
        retrieved: list[RetrievedChunk]
    ) -> float:
        """
        Groundedness (does the answer actually say what the evidence
        says) is a meaningfully different signal than retrieval/rerank
        score (did we find topically relevant documents), and only
        groundedness genuinely reflects confidence in the *answer* - a
        perfect retrieval match can still be paired with a fabricated
        answer. Falls back to the top retrieval score only when no
        groundedness signal exists at all (hallucination guard disabled).
        """
        if groundedness is not None:
            return round(max(0.0, min(groundedness, 1.0)), 4)

        top_retrieval_score = max([item.score for item in retrieved], default=0.0)
        return round(max(0.0, min(top_retrieval_score, 1.0)), 4)

    def _retrieve(
        self,
        query: str,
        top_k: int,
        client_id: str | None = None,
        access_groups: list[str] | None = None
    ) -> list[RetrievedChunk]:
        if self.reranker is None or not self._reranker_enabled_for(client_id):
            candidates = self.retriever.retrieve(
                query=query,
                top_k=top_k
            )
            return self._filter_by_access(candidates, access_groups)[:top_k]

        candidates = self.retriever.retrieve(
            query=query,
            top_k=top_k * self.candidate_multiplier
        )
        authorized = self._filter_by_access(candidates, access_groups)
        return self.reranker.rerank(
            query=query,
            candidates=authorized,
            top_k=top_k
        )

    def _filter_by_access(
        self,
        candidates: list[RetrievedChunk],
        access_groups: list[str] | None
    ) -> list[RetrievedChunk]:
        """
        Excludes unauthorized chunks before they ever reach the reranker
        or the generation prompt - never retrieve-then-hide. A chunk with
        an empty access_groups list (the default - see Chunk's own
        docstring) is accessible to everyone; a chunk with a non-empty
        list is only returned when the caller's own groups intersect it.
        access_groups=None (no authenticated caller / auth disabled)
        behaves the same as an empty list: only unrestricted chunks are
        visible, never a scoped-but-mismatched one.
        """
        caller_groups = set(access_groups or [])
        return [
            candidate
            for candidate in candidates
            if not candidate.chunk.access_groups
            or caller_groups.intersection(candidate.chunk.access_groups)
        ]

    def _reranker_enabled_for(
        self,
        client_id: str | None
    ) -> bool:
        """
        When no FeatureFlagManager is wired in (the default), the reranker
        is used unconditionally whenever configured - unchanged from
        before feature flags existed. When one is wired in (via
        service_factory, FEATURE_FLAGS_ENABLED=true) and no flag by this
        name has been defined yet, that's a caller error rather than a
        silent full rollout - fail open to "reranker enabled" so a missing
        flag definition can't quietly regress retrieval quality for
        everyone.
        """
        if self.feature_flags is None:
            return True

        subject_id = client_id or str(uuid.uuid4())

        try:
            return self.feature_flags.is_enabled_for(RERANKER_FLAG_NAME, subject_id)
        except KeyError:
            return True

    def _is_path_allowed(
        self,
        file_path: str
    ) -> bool:
        """
        No restriction when ingest_allowed_dir isn't set - the default for
        direct construction (tests, scripts, main.py's demo run). When it
        is set (service_factory always sets it for the live API), resolves
        symlinks/".." segments and requires the result to actually sit
        inside that directory, so neither a traversal path
        ("../../etc/passwd") nor an absolute path outside it can reach the
        filesystem through an unauthenticated network endpoint.
        """
        if self.ingest_allowed_dir is None:
            return True

        try:
            resolved = Path(file_path).resolve()
        except (OSError, ValueError):
            return False

        return resolved == self.ingest_allowed_dir or self.ingest_allowed_dir in resolved.parents

    def _format_error(
        self,
        file_path: str,
        error: object
    ) -> str:
        if error is None:
            return f"{file_path}: UNKNOWN_ERROR"

        code = getattr(error, "code", "UNKNOWN_ERROR")
        message = getattr(error, "message", "")
        return f"{file_path}: {code} {message}".strip()
