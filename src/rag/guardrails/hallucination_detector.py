import re

from rag.embeddings.base import Embedder
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity


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
        context_text = " ".join(
            item.chunk.text for item in context.retrieved_chunks
        )
        score = self._groundedness_score(answer, context_text)
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

    def _groundedness_score(
        self,
        answer: str,
        context_text: str
    ) -> float:
        token_overlap = self._token_overlap(answer, context_text)

        if self.embedder is None or not answer.strip() or not context_text.strip():
            return token_overlap

        similarity = self._cosine_similarity(
            self.embedder.embed(answer),
            self.embedder.embed(context_text)
        )
        blended = (
            self.token_overlap_weight * token_overlap
            + self.similarity_weight * similarity
        )
        return max(0.0, min(blended, 1.0))

    def _token_overlap(
        self,
        answer: str,
        context_text: str
    ) -> float:
        answer_terms = self._tokens(answer)

        if not answer_terms:
            return 0.0

        context_terms = self._tokens(context_text)

        if not context_terms:
            return 0.0

        overlap = answer_terms.intersection(context_terms)
        return len(overlap) / len(answer_terms)

    def _tokens(
        self,
        text: str
    ) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _cosine_similarity(
        self,
        first: list[float],
        second: list[float]
    ) -> float:
        numerator = sum(a * b for a, b in zip(first, second))
        first_norm = sum(a * a for a in first) ** 0.5
        second_norm = sum(b * b for b in second) ** 0.5

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return numerator / (first_norm * second_norm)
