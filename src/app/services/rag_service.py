from app.schemas import AskResponse
from app.schemas import IngestResponse
from app.schemas import Source
from ingestion.ingestion_pipeline import IngestionPipeline
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
from rag.vector_store.base import VectorStore
from rag.vector_store.in_memory_store import InMemoryVectorStore


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
        guardrail_manager: GuardrailManager | None = None,
        guardrails_enabled: bool = True,
        pii_guard_enabled: bool = True,
        hallucination_guard_enabled: bool = True,
        groundedness_threshold: float = 0.60
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline or IngestionPipeline()
        self.chunker = chunker or RecursiveChunker()
        self.embedder = embedder or HashingEmbedder()
        self.vector_store = vector_store or InMemoryVectorStore()
        self.answerer = answerer or ExtractiveAnswerer()
        self.reranker = reranker
        self.candidate_multiplier = candidate_multiplier
        self.guardrail_manager = guardrail_manager or (
            GuardrailManager.default(
                embedder=self.embedder,
                pii_enabled=pii_guard_enabled,
                hallucination_enabled=hallucination_guard_enabled,
                groundedness_threshold=groundedness_threshold
            )
            if guardrails_enabled else
            GuardrailManager(guardrails=[])
        )
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

    def ingest(
        self,
        file_paths: list[str]
    ) -> IngestResponse:
        indexed_documents = 0
        indexed_chunks = 0
        errors = []

        for file_path in file_paths:
            document_result = self.ingestion_pipeline.ingest_file(file_path)

            if not document_result.success or document_result.data is None:
                errors.append(self._format_error(file_path, document_result.error))
                continue

            chunk_result = self.chunker.chunk(document_result.data)

            if not chunk_result.success or chunk_result.data is None:
                errors.append(self._format_error(file_path, chunk_result.error))
                continue

            records = [
                (chunk, self.embedder.embed(chunk.text))
                for chunk in chunk_result.data
            ]
            self.vector_store.add_many(records)
            indexed_documents += 1
            indexed_chunks += len(records)

        return IngestResponse(
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            errors=errors
        )

    def ask(
        self,
        query: str,
        top_k: int = 5
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
            top_k=top_k
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
                chunk_id=item.chunk.chunk_id,
                source=item.chunk.source,
                score=item.score,
                text=item.chunk.text
            )
            for item in retrieved
        ]

        confidence = max(
            [item.score for item in retrieved],
            default=0.0
        )

        return AskResponse(
            answer=output_result.text,
            sources=sources,
            confidence=round(max(0.0, min(confidence, 1.0)), 4),
            guardrail_flags=output_result.flags
        )

    def _retrieve(
        self,
        query: str,
        top_k: int
    ) -> list[RetrievedChunk]:
        if self.reranker is None:
            return self.retriever.retrieve(
                query=query,
                top_k=top_k
            )

        candidates = self.retriever.retrieve(
            query=query,
            top_k=top_k * self.candidate_multiplier
        )
        return self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k
        )

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
