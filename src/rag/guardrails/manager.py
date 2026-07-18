import logging
import time
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any

from rag.embeddings.base import Embedder
from rag.guardrails.base import ACTION_RANK
from rag.guardrails.base import Action
from rag.guardrails.base import Guardrail
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.hallucination_detector import HallucinationDetector
from rag.guardrails.pii_guard import PIIGuard
from rag.retrieval.hybrid_retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

BLOCKED_MESSAGE = "This response was blocked by content safety guardrails."


@dataclass(frozen=True)
class GuardrailResult:
    findings: list[GuardrailFinding]
    action: Action
    text: str
    flags: dict[str, Any] = field(default_factory=dict)


class GuardrailManager:
    """
    Runs every registered guardrail for a given pipeline stage, applies
    any redactions in sequence, resolves the strictest action across all
    triggered findings, and (optionally) lets a PolicyEngine escalate
    further. New guardrails register here - RAGService and everything
    downstream is unaffected either way.
    """

    def __init__(
        self,
        guardrails: list[Guardrail] | None = None,
        policy_engine: Any | None = None
    ) -> None:
        self.guardrails = guardrails or []
        self.policy_engine = policy_engine

    @classmethod
    def default(
        cls,
        embedder: Embedder | None = None,
        pii_enabled: bool = True,
        hallucination_enabled: bool = True,
        groundedness_threshold: float = 0.60
    ) -> "GuardrailManager":
        guardrails: list[Guardrail] = []

        if pii_enabled:
            guardrails.append(PIIGuard())

        if hallucination_enabled:
            guardrails.append(
                HallucinationDetector(
                    threshold=groundedness_threshold,
                    embedder=embedder
                )
            )

        return cls(guardrails=guardrails)

    def run_input(
        self,
        query: str
    ) -> GuardrailResult:
        context = GuardrailContext(query=query)
        return self._run(context, GuardrailStage.INPUT, query)

    def run_output(
        self,
        query: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> GuardrailResult:
        context = GuardrailContext(
            query=query,
            answer=answer,
            retrieved_chunks=retrieved_chunks
        )
        return self._run(context, GuardrailStage.OUTPUT, answer)

    def _run(
        self,
        context: GuardrailContext,
        stage: GuardrailStage,
        initial_text: str
    ) -> GuardrailResult:
        findings: list[GuardrailFinding] = []
        current_text = initial_text
        current_context = context

        for guardrail in self.guardrails:
            if guardrail.stage != stage:
                continue

            started_at = time.monotonic()
            finding = guardrail.check(current_context)
            latency_seconds = round(time.monotonic() - started_at, 4)

            if finding.redacted_text is not None:
                current_text = finding.redacted_text
                current_context = self._with_text(current_context, stage, current_text)

            findings.append(finding)
            self._log_finding(finding, stage, latency_seconds)

        action = self._resolve_action(findings)

        if self.policy_engine is not None:
            action = self.policy_engine.evaluate(findings, default_action=action)

        if action == Action.BLOCK:
            current_text = BLOCKED_MESSAGE

        return GuardrailResult(
            findings=findings,
            action=action,
            text=current_text,
            flags=self._build_flags(findings)
        )

    def _with_text(
        self,
        context: GuardrailContext,
        stage: GuardrailStage,
        text: str
    ) -> GuardrailContext:
        if stage == GuardrailStage.INPUT:
            return replace(context, query=text)

        return replace(context, answer=text)

    def _resolve_action(
        self,
        findings: list[GuardrailFinding]
    ) -> Action:
        triggered = [finding.action for finding in findings if finding.triggered]

        if not triggered:
            return Action.ALLOW

        return max(triggered, key=lambda action: ACTION_RANK[action])

    def _build_flags(
        self,
        findings: list[GuardrailFinding]
    ) -> dict[str, Any]:
        flags: dict[str, Any] = {}

        for finding in findings:
            if finding.guardrail_name == "pii_guard":
                flags["pii_detected"] = finding.triggered
            elif finding.guardrail_name == "hallucination_detector":
                flags["hallucination"] = finding.triggered
                if "groundedness_score" in finding.metadata:
                    flags["groundedness"] = finding.metadata["groundedness_score"]
            else:
                flags[finding.guardrail_name] = finding.triggered

        if findings:
            flags["details"] = [
                {
                    "guardrail": finding.guardrail_name,
                    "triggered": finding.triggered,
                    "severity": finding.severity.value,
                    "action": finding.action.value,
                    "message": finding.message,
                    **finding.metadata
                }
                for finding in findings
            ]

        return flags

    def _log_finding(
        self,
        finding: GuardrailFinding,
        stage: GuardrailStage,
        latency_seconds: float
    ) -> None:
        logger.info(
            "guardrail_evaluated",
            extra={
                "guardrail": finding.guardrail_name,
                "stage": stage.value,
                "triggered": finding.triggered,
                "severity": finding.severity.value,
                "action": finding.action.value,
                "latency_seconds": latency_seconds
            }
        )
