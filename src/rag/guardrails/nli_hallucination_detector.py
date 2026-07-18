import logging
import time

from sentence_transformers import CrossEncoder

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

logger = logging.getLogger(__name__)

# cross-encoder/nli-* models label their 3 output classes in this order
# (verified against the model's config.json id2label mapping) - if a
# different NLI model is swapped in, check its own label order.
ENTAILMENT_LABEL_INDEX = 1


class NLIHallucinationDetector:
    """
    Phase 2 example guardrail (output stage): scores groundedness via
    natural language inference instead of token overlap. For each
    retrieved chunk (premise) and the generated answer (hypothesis), a
    cross-encoder NLI model predicts contradiction/entailment/neutral;
    the detector takes the highest entailment probability across chunks
    as the groundedness score - the answer only needs to be entailed by
    at least one chunk to count as grounded. Same sentence-transformers
    CrossEncoder pattern as the reranker: load the model once, reuse it.
    """
    name = "nli_hallucination_detector"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        threshold: float = 0.50
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.model = CrossEncoder(model_name)

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        answer = context.answer or ""

        if not answer.strip() or not context.retrieved_chunks:
            return GuardrailFinding(
                guardrail_name=self.name,
                triggered=True,
                severity=Severity.WARNING,
                action=Action.WARN,
                message="no answer or context to evaluate",
                metadata={"groundedness_score": 0.0, "likely_hallucination": True}
            )

        started_at = time.monotonic()
        pairs = [(item.chunk.text, answer) for item in context.retrieved_chunks]
        probabilities = self.model.predict(pairs, apply_softmax=True)
        score = float(max(row[ENTAILMENT_LABEL_INDEX] for row in probabilities))
        likely_hallucination = score < self.threshold

        logger.info(
            "nli_hallucination_check_completed",
            extra={
                "guardrail": self.name,
                "model": self.model_name,
                "candidate_count": len(pairs),
                "latency_seconds": round(time.monotonic() - started_at, 3)
            }
        )

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=likely_hallucination,
            severity=Severity.WARNING if likely_hallucination else Severity.INFO,
            action=Action.WARN if likely_hallucination else Action.ALLOW,
            message=(
                f"max entailment probability {score:.2f} below threshold {self.threshold:.2f}"
                if likely_hallucination else
                f"max entailment probability {score:.2f} meets threshold {self.threshold:.2f}"
            ),
            metadata={
                "groundedness_score": round(score, 4),
                "likely_hallucination": likely_hallucination
            }
        )
