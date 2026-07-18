import re

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

# deliberately a small, illustrative wordlist - not a real content-safety
# filter. Swap in a maintained list or a classifier-backed implementation
# of the same Guardrail interface for production use.
DEFAULT_BLOCKLIST: frozenset[str] = frozenset({"damn", "hell", "crap"})


class ProfanityGuard:
    """
    Phase 2 example guardrail (output stage): flags answers containing
    words from a configurable blocklist. Exists to demonstrate a third
    category (content quality, vs. PII's privacy or SecretLeakageGuard's
    security) plugging into the same interface with zero new dependencies.
    """
    name = "profanity_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        blocklist: frozenset[str] | None = None
    ) -> None:
        self.blocklist = blocklist or DEFAULT_BLOCKLIST

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        text = context.answer or ""
        words = set(re.findall(r"[a-z']+", text.lower()))
        matched = sorted(words.intersection(self.blocklist))
        triggered = bool(matched)

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=Severity.WARNING if triggered else Severity.INFO,
            action=Action.WARN if triggered else Action.ALLOW,
            message=(
                f"{len(matched)} flagged term(s) found"
                if triggered else "no flagged terms found"
            ),
            metadata={"matched_term_count": len(matched)}
        )
