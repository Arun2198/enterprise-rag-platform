from rag.embeddings.base import Embedder
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.groundedness import blended_groundedness_score
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class HallucinationDetector:
    """
    Lightweight groundedness check: how much of the generated answer's
    vocabulary actually shows up in the retrieved context. When an
    Embedder is available it also blends in embedding cosine similarity as
    a cheap stand-in for "sentence similarity" - RAGService always has one
    (HashingEmbedder by default), so this gets that signal for free with
    no new dependency. Swap in an NLI/BERTScore/RAGAS/LLM-as-judge
    implementation of the same Guardrail interface for a stronger
    production-grade detector; nothing else needs to change.

    Scores the answer against its single best-matching retrieved chunk,
    not all retrieved chunks concatenated - a real dilution bug, found
    live against the deployed app: scoring against the whole concatenated
    top_k context meant a larger top_k (more retrieved chunks, most of
    them on unrelated topics) pulled the score down even when the answer
    was fully grounded in one chunk of it. Same query, same document, same
    correct fact ("Full-time employees accrue 22 days of paid vacation
    leave per calendar year") scored 0.91 groundedness at top_k=3 and 0.58
    at top_k=8 purely from denominator/embedding dilution across the wider
    concatenated context, incorrectly triggering abstention at the larger
    top_k despite retrieval having found the right chunk both times.
    """
    name = "hallucination_detector"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        threshold: float = 0.60,
        embedder: Embedder | None = None,
        token_overlap_weight: float = 0.6,
        similarity_weight: float = 0.4
    ) -> None:
        self.threshold = threshold
        self.embedder = embedder
        self.token_overlap_weight = token_overlap_weight
        self.similarity_weight = similarity_weight

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        answer = context.answer or ""
        score = self._best_chunk_groundedness_score(answer, context.retrieved_chunks)
        likely_hallucination = score < self.threshold

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=likely_hallucination,
            severity=Severity.WARNING if likely_hallucination else Severity.INFO,
            action=Action.WARN if likely_hallucination else Action.ALLOW,
            message=(
                f"groundedness {score:.2f} below threshold {self.threshold:.2f}"
                if likely_hallucination else
                f"groundedness {score:.2f} meets threshold {self.threshold:.2f}"
            ),
            metadata={
                "groundedness_score": round(score, 4),
                "likely_hallucination": likely_hallucination
            }
        )

    def _best_chunk_groundedness_score(
        self,
        answer: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> float:
        """
        Max over individual chunks, not one score against everything
        concatenated - the answer only needs to be grounded in *some* of
        what was retrieved, not all of it at once. A larger top_k pulling
        in more tangential chunks shouldn't be able to drag a genuinely
        grounded answer's score down; see the class docstring for the
        live-reproduced case that motivated this.
        """
        if not retrieved_chunks:
            return 0.0

        return max(
            blended_groundedness_score(
                answer, item.chunk.text, self.embedder,
                self.token_overlap_weight, self.similarity_weight
            )
            for item in retrieved_chunks
        )
