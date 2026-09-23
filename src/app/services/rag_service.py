import dataclasses
import logging
import time
import uuid
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
from pathlib import Path

from app.schemas import AskResponse
from app.schemas import CitationResponse
from app.schemas import IngestResponse
from app.schemas import Source
from ingestion.contracts.document import Document
from ingestion.incremental_indexer import IncrementalIndexer
from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.manifest_store import ManifestStore
from mlops.feature_flags import FeatureFlagManager
from rag.chunking.chunk import Chunk
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.base import Embedder
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.generation.base import Answerer
from rag.generation.citations import extract_citations
from rag.generation.document_first_answerer import DocumentFirstAnswerer
from rag.generation.extractive_answerer import ExtractiveAnswerer
from rag.generation.prompt import ConversationTurn
from rag.guardrails.base import Action
from rag.guardrails.groundedness import blended_groundedness_score
from rag.guardrails.manager import GuardrailManager
from rag.retrieval.hybrid_retrieval import HybridRetriever
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.reranker import CrossEncoderReranker
from rag.retrieval.trace import CandidateTrace
from rag.retrieval.trace import RetrievalTrace
from rag.vector_store.base import VectorStore
from rag.vector_store.in_memory_store import InMemoryVectorStore
from rag.vector_store.in_memory_store import MetadataFilter

logger = logging.getLogger(__name__)

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
        abstention_enabled: bool = True,
        manifest_store: ManifestStore | None = None,
        grounded_first_enabled: bool = True,
        grounded_first_threshold: float = 0.60
    ) -> None:
        self.abstention_enabled = abstention_enabled
        self.ingest_allowed_dir = (
            Path(ingest_allowed_dir).resolve() if ingest_allowed_dir is not None else None
        )
        self.ingestion_pipeline = ingestion_pipeline or IngestionPipeline()
        self.chunker = chunker or RecursiveChunker()
        self.embedder = embedder or HashingEmbedder()
        # `or` would be wrong here: InMemoryVectorStore defines __len__,
        # so a caller-supplied but still-empty store is falsy and `or`
        # would silently discard it in favor of a brand-new one - found
        # via a real test that passed an empty store in and then
        # asserted against that exact reference. An explicit None check
        # is the only correct way to default an object that can be
        # "present but empty".
        self.vector_store = vector_store if vector_store is not None else InMemoryVectorStore()
        self.answerer = answerer or ExtractiveAnswerer()
        self.grounded_first_enabled = grounded_first_enabled
        self.grounded_first_threshold = grounded_first_threshold
        # Dedicated instance, always available regardless of what
        # self.answerer is configured as (an LLM-only provider has no
        # extractive fallback of its own to reach for) - see
        # _answer_grounded_first().
        self._grounded_first_answerer = ExtractiveAnswerer()
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
        # None (the default for direct construction - tests/scripts/
        # main.py's demo run) keeps index_document()'s old behavior
        # completely unchanged: full delete-free re-chunk + re-embed +
        # add_many of every chunk, every call. service_factory wires a
        # real ManifestStore in for the live app, which is what actually
        # turns on incremental page/chunk-level re-embedding - see
        # IncrementalIndexer.
        self.manifest_store = manifest_store
        self.incremental_indexer = (
            IncrementalIndexer(
                chunker=self.chunker,
                embedder=self.embedder,
                vector_store=self.vector_store,
                manifest_store=self.manifest_store
            )
            if self.manifest_store is not None else None
        )

    def ingest(
        self,
        file_paths: list[str],
        document_ids: Sequence[str]
    ) -> IngestResponse:
        """
        document_ids is mandatory and must be the same length as
        file_paths - entry i is the real stable identity for
        file_paths[i]. There is no filename-derived fallback: every
        caller must track and supply its own document_id, on every
        ingest of the same document (including after a rename), so a
        rename is never mistaken for a brand-new document and
        incremental re-embedding (IncrementalIndexer, which diffs purely
        on document_id) keeps working across one.
        """
        indexed_documents = 0
        indexed_chunks = 0
        errors = []

        if len(document_ids) != len(file_paths):
            return IngestResponse(
                indexed_documents=0,
                indexed_chunks=0,
                errors=[
                    "DOCUMENT_IDS_LENGTH_MISMATCH document_ids must have the same "
                    "length as file_paths"
                ]
            )

        for index, file_path in enumerate(file_paths):
            if not self._is_path_allowed(file_path):
                errors.append(
                    f"{file_path}: PATH_NOT_ALLOWED file path is outside the allowed "
                    "ingestion directory"
                )
                continue

            document_result = self.ingestion_pipeline.ingest_file(file_path, document_id=document_ids[index])

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

        When a manifest_store is configured, delegates to
        IncrementalIndexer, which embeds and upserts only pages/chunks
        whose content actually changed since the last ingest of this
        document_id (and deletes chunks for pages/content that no longer
        exist) instead of re-embedding everything unconditionally.
        """
        if self.incremental_indexer is not None:
            result = self.incremental_indexer.index(document)
            return result.indexed_chunk_count if result is not None else None

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
        records = list(zip(chunks, embeddings, strict=True))
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

        Also clears the document's manifest (when incremental indexing is
        on) - otherwise a later re-upload of the same document_id would
        see a stale manifest claiming chunks are already indexed and
        skip re-embedding content that was just deleted.
        """
        deleted = self.vector_store.delete_by_document(document_id)

        if self.manifest_store is not None:
            self.manifest_store.delete(document_id)

        return deleted

    def reindex_document(
        self,
        file_path: str,
        document_id: str
    ) -> IngestResponse:
        """
        Delete-then-reingest for a document that changed, when incremental
        indexing isn't on (manifest_store=None) - not a partial update, a
        full replace, since chunk boundaries/count can shift with any
        content change and stale chunks from the old version must not
        survive alongside the new ones.

        When a manifest_store is configured, skips the explicit delete:
        index_document()'s IncrementalIndexer path already does the
        correct diff-based delete of stale chunks (and upsert of
        changed/new ones) using deterministic page/chunk-scoped ids, so
        deleting everything first would only throw away the manifest
        history that makes the diff possible and force a full re-embed -
        exactly the behavior this whole feature exists to avoid.

        document_id is mandatory, same as ingest() - no filename-derived
        fallback, so the caller must supply the file's real stable
        identity, including after a rename, for this to be recognized as
        the same document rather than a new one.
        """
        document_result = self.ingestion_pipeline.ingest_file(file_path, document_id=document_id)

        if not document_result.success or document_result.data is None:
            return IngestResponse(
                indexed_documents=0,
                indexed_chunks=0,
                errors=[self._format_error(file_path, document_result.error)]
            )

        if self.incremental_indexer is None:
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
        access_groups: list[str] | None = None,
        history: list[ConversationTurn] | None = None,
        document_ids: list[str] | None = None
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
            access_groups=access_groups,
            document_ids=document_ids
        )
        answer, _ = self._answer(query, retrieved, history)

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

        citations = extract_citations(answer_text, retrieved)
        flags = output_result.flags

        if citations:
            flags = {**flags, "has_invalid_citations": any(not c.valid for c in citations)}

        return AskResponse(
            answer=answer_text,
            sources=sources,
            groundedness=groundedness,
            confidence=confidence,
            citations=[CitationResponse(**vars(c)) for c in citations],
            guardrail_flags=flags
        )

    def ask_with_trace(
        self,
        query: str,
        top_k: int = 5,
        client_id: str | None = None,
        access_groups: list[str] | None = None,
        history: list[ConversationTurn] | None = None,
        document_ids: list[str] | None = None
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
            access_groups=access_groups,
            document_ids=document_ids
        )

        generation_started = time.monotonic()
        answer, provider_used = self._answer(query, retrieved, history)
        trace.stage_timings_ms["generation"] = (time.monotonic() - generation_started) * 1000
        trace.generation_provider = provider_used
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

        if self.abstention_enabled and self._should_abstain(output_result.flags):
            answer_text = ABSTENTION_MESSAGE

        citations = extract_citations(answer_text, retrieved)
        flags = output_result.flags

        if citations:
            flags = {**flags, "has_invalid_citations": any(not c.valid for c in citations)}

        return AskResponse(
            answer=answer_text,
            sources=sources,
            groundedness=groundedness,
            confidence=confidence,
            citations=[CitationResponse(**vars(c)) for c in citations],
            guardrail_flags=flags
        ), trace

    def _retrieve_with_trace(
        self,
        query: str,
        top_k: int,
        client_id: str | None = None,
        access_groups: list[str] | None = None,
        document_ids: list[str] | None = None
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        metadata_filter = self._document_scope_filter(document_ids)

        if self.reranker is None or not self._reranker_enabled_for(client_id):
            candidates, trace = self.retriever.retrieve_with_trace(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
            return self._filter_by_access(candidates, access_groups)[:top_k], trace

        candidates, trace = self.retriever.retrieve_with_trace(
            query=query,
            top_k=top_k * self.candidate_multiplier,
            metadata_filter=metadata_filter
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
        access_groups: list[str] | None = None,
        document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        metadata_filter = self._document_scope_filter(document_ids)

        if self.reranker is None or not self._reranker_enabled_for(client_id):
            candidates = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter
            )
            return self._filter_by_access(candidates, access_groups)[:top_k]

        candidates = self.retriever.retrieve(
            query=query,
            top_k=top_k * self.candidate_multiplier,
            metadata_filter=metadata_filter
        )
        authorized = self._filter_by_access(candidates, access_groups)
        return self.reranker.rerank(
            query=query,
            candidates=authorized,
            top_k=top_k
        )

    def _expand_to_window(
        self,
        retrieved: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """
        Expands each sentence-level match to its parent window (the
        sentence-packed group RecursiveChunker split it from) before
        generation sees it - a lone sentence often can't carry enough
        context on its own for the LLM to answer well, even though it's
        exactly the right unit for precise retrieval/citation. Only
        used for what gets handed to the Answerer; sources/citations/
        guardrail scoring still reference the original, precise
        sentence-level chunks (see ask()/ask_with_trace()) - a source
        should point at the sentence that actually matched, not the
        whole paragraph around it.

        A chunk with no parent_chunk_id (pre-sentence-granularity data
        still in the index, or a store that returns nothing for the
        lookup) passes through unchanged rather than erroring.
        """
        expanded: list[RetrievedChunk] = []

        for item in retrieved:
            parent_chunk_id = item.chunk.parent_chunk_id

            if parent_chunk_id is None:
                expanded.append(item)
                continue

            try:
                siblings = self.vector_store.get_by_parent_chunk_id(parent_chunk_id)
            except Exception:
                siblings = []

            if not siblings:
                expanded.append(item)
                continue

            window_text = " ".join(
                sibling.text
                for sibling in sorted(siblings, key=self._sentence_index_of)
            )
            expanded.append(
                dataclasses.replace(item, chunk=item.chunk.model_copy(update={"text": window_text}))
            )

        return expanded

    def _sentence_index_of(
        self,
        chunk: Chunk
    ) -> int:
        # chunk_id is "{document_id}:p{page}:s{sentence_index}" -
        # extracting the trailing integer reconstructs each window's
        # original sentence order without needing a separate index.
        try:
            return int(chunk.chunk_id.rsplit(":", 1)[-1].removeprefix("s"))
        except ValueError:
            return 0

    def _answer(
        self,
        query: str,
        retrieved: list[RetrievedChunk],
        history: list[ConversationTurn] | None
    ) -> tuple[str, str]:
        """
        Grounded-first answering: decide, before ever calling an LLM,
        whether the retrieved context alone is confident enough to
        answer directly - genuinely distinct from FallbackAnswerer,
        which only reacts to a provider *raising* (a failure), never to
        answer *quality*. Both can be active at once: this decides
        whether self.answerer gets called at all; if self.answerer is a
        FallbackAnswerer, it still handles what happens if that call
        fails, exactly as before this existed.

        Returns (answer_text, provider_name_used) - the second value is
        purely for ask_with_trace()'s generation_provider field, so the
        debug trace reports which answerer actually ran, not just
        whichever was configured.

        Falls straight through to self.answerer (unchanged from before
        grounded-first routing existed) when:
        - grounded_first_enabled is False
        - nothing was retrieved - no context to be confident about
        - self.answerer is already ExtractiveAnswerer - already
          document-only, nothing to route away from
        - self.answerer is already a DocumentFirstAnswerer - that class
          already does its own retrieval-confidence routing (a
          different signal - query/chunk cosine similarity rather than
          this blended groundedness score); stacking a second gate on
          top would just double-decide the same question with two
          different answers possible.
        """
        expanded = self._expand_to_window(retrieved)

        if (
            not self.grounded_first_enabled
            or not retrieved
            or isinstance(self.answerer, ExtractiveAnswerer)
            or isinstance(self.answerer, DocumentFirstAnswerer)
        ):
            answer = self.answerer.answer(query=query, retrieved_chunks=expanded, history=history)
            return answer, type(self.answerer).__name__

        confidence = self._retrieval_confidence(query, expanded)
        route_to_extractive = confidence >= self.grounded_first_threshold

        logger.info(
            "grounded_first_routed",
            extra={
                "route": "extractive" if route_to_extractive else "llm",
                "confidence": round(confidence, 4),
                "threshold": self.grounded_first_threshold
            }
        )

        if route_to_extractive:
            answer = self._grounded_first_answerer.answer(query=query, retrieved_chunks=expanded, history=history)
            return answer, type(self._grounded_first_answerer).__name__

        answer = self.answerer.answer(query=query, retrieved_chunks=expanded, history=history)
        return answer, type(self.answerer).__name__

    def _retrieval_confidence(
        self,
        query: str,
        retrieved: list[RetrievedChunk]
    ) -> float:
        """
        Reuses HallucinationDetector's own blended groundedness scoring
        (token overlap blended with embedding cosine similarity, when
        an embedder is available) - applied to (query, chunk text)
        instead of (answer, chunk text). Same underlying question ("how
        well does this text cover that text"), just asked before
        generation rather than after. Max over individual chunks, not
        concatenated, for the same reason HallucinationDetector scores
        that way - a larger top_k pulling in more tangential chunks
        shouldn't be able to drag a genuinely well-covered query's
        score down.
        """
        return max(
            blended_groundedness_score(query, item.chunk.text, self.embedder)
            for item in retrieved
        )

    def _document_scope_filter(
        self,
        document_ids: list[str] | None
    ) -> MetadataFilter | None:
        """
        Builds the metadata_filter passed to HybridRetriever.retrieve()
        when a caller wants a query scoped to specific documents.
        document_id is already mirrored into every chunk's metadata dict
        by RecursiveChunker, so this needs no new indexing - both
        InMemoryVectorStore and OpenSearchVectorStore apply the filter
        inside the search itself (before ranking), not as a post-hoc
        filter on already-ranked results, so a scoped query doesn't
        waste its top_k budget on chunks from excluded documents.
        None (the default - no document_ids given) means unscoped,
        unchanged from before this existed.
        """
        if not document_ids:
            return None

        return {"document_id": document_ids}

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
