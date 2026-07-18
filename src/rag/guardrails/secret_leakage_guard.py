import re
from re import Pattern

from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

DEFAULT_PATTERNS: dict[str, Pattern[str]] = {
    "AWS_ACCESS_KEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GITHUB_TOKEN": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "PRIVATE_KEY_BLOCK": re.compile(r"-----BEGIN (RSA|EC|OPENSSH|DSA|PGP)? ?PRIVATE KEY-----"),
    "BEARER_TOKEN": re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*"),
    "GENERIC_API_KEY": re.compile(
        r"\b(?:api[_-]?key|apikey)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?",
        re.IGNORECASE
    ),
    "PASSWORD_ASSIGNMENT": re.compile(
        r"\bpassword\s*[:=]\s*['\"]?\S{6,}['\"]?",
        re.IGNORECASE
    ),
}


class SecretLeakageGuard:
    """
    Phase 2 example guardrail (output stage): regex detection/redaction
    for credentials that shouldn't appear in a generated answer (API
    keys, tokens, private key blocks, password assignments). Same
    Guardrail interface as PIIGuard - this exists to prove new guardrails
    plug in without touching RAGService, not as an exhaustive secret
    scanner.
    """
    name = "secret_leakage_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        patterns: dict[str, Pattern[str]] | None = None
    ) -> None:
        self.patterns = patterns or DEFAULT_PATTERNS

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        text = context.answer or ""
        redacted = text
        detected: list[str] = []

        for entity, pattern in self.patterns.items():
            if not pattern.search(redacted):
                continue

            detected.append(entity)
            redacted = pattern.sub(f"[REDACTED_{entity}]", redacted)

        triggered = bool(detected)

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=triggered,
            severity=Severity.CRITICAL if triggered else Severity.INFO,
            action=Action.BLOCK if triggered else Action.ALLOW,
            message=(
                f"{len(detected)} secret pattern(s) detected: {', '.join(detected)}"
                if triggered else "no secrets detected"
            ),
            redacted_text=redacted if triggered else None,
            metadata={"detected_entities": detected}
        )
