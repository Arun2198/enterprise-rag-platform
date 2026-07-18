import re
from re import Pattern

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

DEFAULT_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"ignore (all |any )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"disregard (all |any )?(previous|prior|above) (instructions|rules)", re.IGNORECASE),
    re.compile(r"you are now (in )?(developer|dan|jailbreak) mode", re.IGNORECASE),
    re.compile(r"pretend (you are|to be) .*(no rules|unrestricted|without restrictions)", re.IGNORECASE),
    re.compile(r"reveal (your |the )?(system prompt|system instructions)", re.IGNORECASE),
    re.compile(r"act as (if )?(you (have|had) no|there (are|were) no) (restrictions|rules|guardrails)", re.IGNORECASE),
    re.compile(r"\bDAN\b"),
    re.compile(r"bypass (your |the )?(safety|content) (filters?|guardrails?)", re.IGNORECASE),
)


class PromptInjectionGuard:
    """
    Phase 2 example guardrail (input stage): heuristic pattern matching for
    common prompt-injection and jailbreak phrasing in the user query.
    This is a starting point, not a substitute for a trained classifier -
    swap in a model-backed implementation of the same Guardrail interface
    for production use; nothing else needs to change.
    """
    name = "prompt_injection_guard"
    stage = GuardrailStage.INPUT

    def __init__(
        self,
        patterns: tuple[Pattern[str], ...] | None = None
    ) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        query = context.query or ""
        matched = [pattern.pattern for pattern in self.patterns if pattern.search(query)]
        triggered = bool(matched)

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=Severity.HIGH if triggered else Severity.INFO,
            action=Action.BLOCK if triggered else Action.ALLOW,
            message=(
                f"{len(matched)} injection/jailbreak pattern(s) matched"
                if triggered else "no injection patterns matched"
            ),
            metadata={"matched_pattern_count": len(matched)}
        )
