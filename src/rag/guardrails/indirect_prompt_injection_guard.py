from re import Pattern

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.injection_patterns import DEFAULT_PATTERNS


class IndirectPromptInjectionGuard:
    """
    Output-stage detection for injection phrasing found in the RETRIEVED
    CONTEXT itself, not the user's query - PromptInjectionGuard only ever
    looks at context.query, so a malicious instruction planted inside an
    indexed document (the actual "indirect" injection vector) sails past
    it undetected today.

    This runs after generation, so it's a detection/defense-in-depth
    backstop, not prevention - the primary defense is
    rag.generation.prompt.build_grounded_prompt()'s explicit
    evidence-not-instruction framing, which reduces (does not eliminate)
    the chance the model acts on injected content in the first place. If
    this guard triggers, at minimum the suspicious retrieval is flagged
    for review; escalate to BLOCK via PolicyEngine if that's the desired
    response for this deployment. Prompt injection is not claimed to be
    fully solved by either layer.
    """
    name = "indirect_prompt_injection_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        patterns: tuple[Pattern[str], ...] | None = None
    ) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        matched_chunk_ids: list[str] = []

        for item in context.retrieved_chunks:
            if any(pattern.search(item.chunk.text) for pattern in self.patterns):
                matched_chunk_ids.append(item.chunk.chunk_id)

        triggered = bool(matched_chunk_ids)

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=Severity.HIGH if triggered else Severity.INFO,
            action=Action.WARN if triggered else Action.ALLOW,
            message=(
                f"{len(matched_chunk_ids)} retrieved chunk(s) contain "
                "injection-like phrasing"
                if triggered else "no injection patterns found in retrieved context"
            ),
            metadata={"matched_chunk_ids": matched_chunk_ids}
        )
