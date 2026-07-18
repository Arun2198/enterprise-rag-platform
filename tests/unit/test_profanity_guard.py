from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailStage
from rag.guardrails.profanity_guard import ProfanityGuard


def test_guard_is_registered_for_output_stage():

    guard = ProfanityGuard()

    assert guard.stage == GuardrailStage.OUTPUT


def test_clean_answer_passes():

    guard = ProfanityGuard()
    context = GuardrailContext(query="q", answer="Everything looks good here.")

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.action == Action.ALLOW


def test_blocklisted_term_is_flagged():

    guard = ProfanityGuard()
    context = GuardrailContext(query="q", answer="Well, that was a hell of a bug.")

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.action == Action.WARN
    assert finding.metadata["matched_term_count"] == 1


def test_custom_blocklist_is_configurable():

    guard = ProfanityGuard(blocklist=frozenset({"widget"}))
    context = GuardrailContext(query="q", answer="This widget is broken.")

    finding = guard.check(context)

    assert finding.triggered is True
