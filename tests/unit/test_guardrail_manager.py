from rag.chunking.chunk import Chunk
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity
from rag.guardrails.manager import BLOCKED_MESSAGE
from rag.guardrails.manager import GuardrailManager
from rag.guardrails.policy import PolicyEngine
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class FakeGuardrail:

    def __init__(self, name, stage, finding):
        self.name = name
        self.stage = stage
        self._finding = finding
        self.calls = 0

    def check(self, context):
        self.calls += 1
        return self._finding


def _make_retrieved_chunk(text: str = "context text") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text=text,
        source="doc.md",
        document_type="markdown"
    )
    return RetrievedChunk(chunk=chunk, vector_score=0.5, keyword_score=0.5, score=0.5)


def test_clean_answer_passes_through_default_manager():

    manager = GuardrailManager.default()

    result = manager.run_output(
        query="q",
        answer="A perfectly ordinary answer.",
        retrieved_chunks=[_make_retrieved_chunk("A perfectly ordinary answer.")]
    )

    assert result.action == Action.ALLOW
    assert result.text == "A perfectly ordinary answer."
    assert result.flags["pii_detected"] is False


def test_pii_gets_redacted_through_manager():

    manager = GuardrailManager.default()

    result = manager.run_output(
        query="q",
        answer="Contact john@company.com for help.",
        retrieved_chunks=[_make_retrieved_chunk("Contact john@company.com for help.")]
    )

    assert result.action == Action.REDACT
    assert "[REDACTED_EMAIL]" in result.text
    assert result.flags["pii_detected"] is True


def test_hallucination_is_flagged_through_manager():

    manager = GuardrailManager.default(groundedness_threshold=0.6)

    result = manager.run_output(
        query="q",
        answer="Completely unrelated statement about astronomy.",
        retrieved_chunks=[_make_retrieved_chunk("Contractors receive 10 days of leave.")]
    )

    assert result.flags["hallucination"] is True
    assert "groundedness" in result.flags


def test_multiple_guardrails_all_execute():

    pii = FakeGuardrail(
        "pii_guard",
        GuardrailStage.OUTPUT,
        GuardrailFinding(
            guardrail_name="pii_guard",
            triggered=False,
            severity=Severity.INFO,
            action=Action.ALLOW,
            message="clean"
        )
    )
    hallucination = FakeGuardrail(
        "hallucination_detector",
        GuardrailStage.OUTPUT,
        GuardrailFinding(
            guardrail_name="hallucination_detector",
            triggered=False,
            severity=Severity.INFO,
            action=Action.ALLOW,
            message="grounded",
            metadata={"groundedness_score": 0.9}
        )
    )
    manager = GuardrailManager(guardrails=[pii, hallucination])

    manager.run_output(query="q", answer="answer", retrieved_chunks=[])

    assert pii.calls == 1
    assert hallucination.calls == 1


def test_disabled_guardrails_are_skipped():

    manager = GuardrailManager.default(pii_enabled=False, hallucination_enabled=True)

    result = manager.run_output(
        query="q",
        answer="Contact john@company.com for help.",
        retrieved_chunks=[_make_retrieved_chunk("Contact john@company.com for help.")]
    )

    assert "pii_detected" not in result.flags
    assert "john@company.com" in result.text


def test_input_stage_guardrails_do_not_run_on_output():

    input_only = FakeGuardrail(
        "input_only",
        GuardrailStage.INPUT,
        GuardrailFinding(
            guardrail_name="input_only",
            triggered=True,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            message="blocked"
        )
    )
    manager = GuardrailManager(guardrails=[input_only])

    result = manager.run_output(query="q", answer="answer", retrieved_chunks=[])

    assert input_only.calls == 0
    assert result.action == Action.ALLOW
    assert result.text == "answer"


def test_severity_is_computed_as_max_across_triggered_findings():

    low = FakeGuardrail(
        "low",
        GuardrailStage.OUTPUT,
        GuardrailFinding(
            guardrail_name="low",
            triggered=True,
            severity=Severity.WARNING,
            action=Action.WARN,
            message="warn"
        )
    )
    high = FakeGuardrail(
        "high",
        GuardrailStage.OUTPUT,
        GuardrailFinding(
            guardrail_name="high",
            triggered=True,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            message="block"
        )
    )
    manager = GuardrailManager(guardrails=[low, high])

    result = manager.run_output(query="q", answer="answer", retrieved_chunks=[])

    assert result.action == Action.BLOCK
    assert result.text == BLOCKED_MESSAGE


def test_policy_engine_can_escalate_action():

    warn_finding = GuardrailFinding(
        guardrail_name="pii_guard",
        triggered=True,
        severity=Severity.HIGH,
        action=Action.REDACT,
        message="redacted"
    )
    pii = FakeGuardrail("pii_guard", GuardrailStage.OUTPUT, warn_finding)
    manager = GuardrailManager(
        guardrails=[pii],
        policy_engine=PolicyEngine.default_policies()
    )

    result = manager.run_output(query="q", answer="sensitive answer", retrieved_chunks=[])

    # default policy: PII + HIGH severity -> BLOCK, overriding the guard's
    # own suggested REDACT action
    assert result.action == Action.BLOCK
    assert result.text == BLOCKED_MESSAGE
