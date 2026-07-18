from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.secret_leakage_guard import SecretLeakageGuard


def test_guard_is_registered_for_output_stage():

    guard = SecretLeakageGuard()

    assert guard.stage == GuardrailStage.OUTPUT


def test_clean_answer_passes():

    guard = SecretLeakageGuard()
    context = GuardrailContext(query="q", answer="The API is stable and documented.")

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.action == Action.ALLOW


def test_aws_access_key_is_redacted():

    guard = SecretLeakageGuard()
    context = GuardrailContext(
        query="q",
        answer="Use key AKIAABCDEFGHIJKLMNOP to authenticate."
    )

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.severity == Severity.CRITICAL
    assert finding.action == Action.BLOCK
    assert "[REDACTED_AWS_ACCESS_KEY]" in finding.redacted_text
    assert "AKIAABCDEFGHIJKLMNOP" not in finding.redacted_text


def test_private_key_block_is_redacted():

    guard = SecretLeakageGuard()
    context = GuardrailContext(
        query="q",
        answer="Here is the key:\n-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJ..."
    )

    finding = guard.check(context)

    assert finding.triggered is True
    assert "PRIVATE_KEY_BLOCK" in finding.metadata["detected_entities"]
    assert "-----BEGIN RSA PRIVATE KEY-----" not in finding.redacted_text
