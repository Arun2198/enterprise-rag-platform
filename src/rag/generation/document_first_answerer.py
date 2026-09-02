import logging

from rag.embeddings.base import Embedder
from rag.generation.base import Answerer
from rag.generation.prompt import ConversationTurn
from rag.guardrails.retrieval_relevance_guard import default_retrieval_relevance_threshold
from rag.retrieval.hybrid_retrieval import RetrievedChunk
from rag.retrieval.relevance import best_retrieval_relevance

logger = logging.getLogger(__name__)


class DocumentFirstAnswerer:
    """
    Retrieval-gated routing between two Answerers: prefer answering
    directly from the retrieved documents, and only fall back to LLM
    generation when retrieval doesn't have a good enough match for the
    query. "Good enough" is the same cosine-similarity signal
    RetrievalRelevanceGuard already uses (query vs. best-matching
    retrieved chunk, both freshly embedded) - reused rather than
    reinvented, and subject to the exact same per-embedder calibration
    caveats documented there (re-verify the threshold if
    EMBEDDING_PROVIDER/EMBEDDING_MODEL_NAME changes).

    This does not add a new "generate freely with no grounding" path:
    llm_answerer still only ever sees the same retrieved_chunks
    HybridRetriever found, and still goes through the same grounded
    prompt template (rag.generation.prompt.build_grounded_prompt) as
    every other LLM-backed Answerer in this project. When retrieval
    finds nothing relevant at all, this falls through to
    llm_answerer.answer() with those (weak or empty) chunks - for
    BedrockAnswerer/OpenAICompatibleAnswerer that means their own
    existing "no retrieved chunks" fallback (a fixed message, no LLM
    call) applies unchanged; this class deliberately does not
    special-case "call the LLM with zero context and let it answer from
    general knowledge" - that would mean regenerating an answer with
    nothing for HallucinationDetector/IndirectPromptInjectionGuard to
    ground against, undermining the guardrail pipeline's whole premise.
    A caller that genuinely wants ungrounded general-knowledge fallback
    needs a deliberately different, explicitly-labeled Answerer - not
    implemented here.
    """

    def __init__(
        self,
        document_answerer: Answerer,
        llm_answerer: Answerer,
        embedder: Embedder,
        threshold: float | None = None
    ) -> None:
        self.document_answerer = document_answerer
        self.llm_answerer = llm_answerer
        self.embedder = embedder
        self.threshold = (
            threshold if threshold is not None
            else default_retrieval_relevance_threshold(embedder)
        )

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        history: list[ConversationTurn] | None = None
    ) -> str:
        if not retrieved_chunks or not query.strip():
            logger.info("document_first_answerer_routed", extra={"route": "llm", "reason": "no_chunks_or_query"})
            return self.llm_answerer.answer(query, retrieved_chunks, history=history)

        relevance = best_retrieval_relevance(self.embedder, query, retrieved_chunks)
        route = "document" if relevance >= self.threshold else "llm"

        logger.info(
            "document_first_answerer_routed",
            extra={"route": route, "relevance": round(relevance, 4), "threshold": self.threshold}
        )

        if route == "document":
            return self.document_answerer.answer(query, retrieved_chunks, history=history)

        return self.llm_answerer.answer(query, retrieved_chunks, history=history)
