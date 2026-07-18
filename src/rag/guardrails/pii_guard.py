import re
from re import Pattern

from rag.guardrails.base import SEVERITY_RANK
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

DEFAULT_PATTERNS: dict[str, Pattern[str]] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "AADHAAR": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
}

# most distinctive patterns first, so a looser one (PHONE) never eats
# digits that already matched a more specific entity (SSN/credit
# card/Aadhaar) earlier in the same pass.
DEFAULT_ENTITY_ORDER: tuple[str, ...] = ("EMAIL", "SSN", "CREDIT_CARD", "AADHAAR", "PHONE")

DEFAULT_SEVERITY: dict[str, Severity] = {
    "EMAIL": Severity.INFO,
    "PHONE": Severity.WARNING,
    "SSN": Severity.HIGH,
    "CREDIT_CARD": Severity.CRITICAL,
    "AADHAAR": Severity.HIGH,
}


class PIIGuard:
    """
    Regex-based PII detector/redactor. Entities are individually
    configurable (patterns, severities, which ones are enabled) so a
    deployment can turn off region-specific identifiers like Aadhaar
    without touching code.
    """
    name = "pii_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        patterns: dict[str, Pattern[str]] | None = None,
        entity_order: tuple[str, ...] | None = None,
        enabled_entities: set[str] | None = None,
        severities: dict[str, Severity] | None = None
    ) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS
        self.entity_order = entity_order or DEFAULT_ENTITY_ORDER
        self.enabled_entities = enabled_entities or set(self.patterns)
        self.severities = severities or DEFAULT_SEVERITY

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        text = context.answer or ""
        redacted = text
        detected: list[str] = []

        for entity in self.entity_order:
            if entity not in self.enabled_entities:
                continue

            pattern = self.patterns.get(entity)

            if pattern is None or not pattern.search(redacted):
                continue

            detected.append(entity)
            redacted = pattern.sub(f"[REDACTED_{entity}]", redacted)

        triggered = bool(detected)

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=self._max_severity(detected),
            action=Action.REDACT if triggered else Action.ALLOW,
            message=(
                f"redacted {len(detected)} PII entity type(s): {', '.join(detected)}"
                if triggered else "no PII detected"
            ),
            redacted_text=redacted if triggered else None,
            metadata={"detected_entities": detected}
        )

    def _max_severity(
        self,
        detected: list[str]
    ) -> Severity:
        if not detected:
            return Severity.INFO

        return max(
            (self.severities.get(entity, Severity.WARNING) for entity in detected),
            key=lambda severity: SEVERITY_RANK[severity]
        )
