from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import Severity
from rag.guardrails.pii_guard import PIIGuard


def test_clean_answer_is_not_triggered():

    guard = PIIGuard()
    context = GuardrailContext(query="q", answer="The office is open 9 to 5.")

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.action == Action.ALLOW
    assert finding.severity == Severity.INFO
    assert finding.redacted_text is None


def test_email_is_redacted():

    guard = PIIGuard()
    context = GuardrailContext(
        query="q",
        answer="Contact john@company.com for details."
    )

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.action == Action.REDACT
    assert "[REDACTED_EMAIL]" in finding.redacted_text
    assert "john@company.com" not in finding.redacted_text
    assert "EMAIL" in finding.metadata["detected_entities"]


def test_ssn_is_redacted_with_high_severity():

    guard = PIIGuard()
    context = GuardrailContext(query="q", answer="SSN on file: 123-45-6789.")

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.severity == Severity.HIGH
    assert "[REDACTED_SSN]" in finding.redacted_text
    assert "123-45-6789" not in finding.redacted_text


def test_credit_card_is_redacted_with_critical_severity():

    guard = PIIGuard()
    context = GuardrailContext(
        query="q",
        answer="Card number 4111111111111111 was charged."
    )

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.severity == Severity.CRITICAL
    assert "[REDACTED_CREDIT_CARD]" in finding.redacted_text
    assert "4111111111111111" not in finding.redacted_text


def test_multiple_entities_produce_max_severity():

    guard = PIIGuard()
    context = GuardrailContext(
        query="q",
        answer="Email me at jane@example.com or call 415-555-0134."
    )

    finding = guard.check(context)

    assert finding.triggered is True
    assert set(finding.metadata["detected_entities"]) >= {"EMAIL", "PHONE"}
    # PHONE (WARNING) should win over EMAIL (INFO)
    assert finding.severity == Severity.WARNING


def test_disabling_an_entity_skips_it():

    guard = PIIGuard(enabled_entities={"SSN", "CREDIT_CARD", "AADHAAR", "PHONE"})
    context = GuardrailContext(
        query="q",
        answer="Contact john@company.com for details."
    )

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.redacted_text is None
