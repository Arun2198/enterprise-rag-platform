from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import Severity
from rag.guardrails.policy import PolicyCondition
from rag.guardrails.policy import PolicyEngine
from rag.guardrails.policy import PolicyRule


def _pii_finding(severity: Severity, triggered: bool = True) -> GuardrailFinding:
    return GuardrailFinding(
        guardrail_name="pii_guard",
        triggered=triggered,
        severity=severity,
        action=Action.REDACT,
        message="pii finding"
    )


def _hallucination_finding(groundedness_score: float) -> GuardrailFinding:
    return GuardrailFinding(
        guardrail_name="hallucination_detector",
        triggered=groundedness_score < 0.6,
        severity=Severity.WARNING,
        action=Action.WARN,
        message="hallucination finding",
        metadata={"groundedness_score": groundedness_score}
    )


def test_condition_requires_matching_guardrail_present():

    condition = PolicyCondition(guardrail_name="pii_guard")

    assert condition.matches([]) is False


def test_condition_min_severity_gate():

    condition = PolicyCondition(guardrail_name="pii_guard", min_severity=Severity.HIGH)

    assert condition.matches([_pii_finding(Severity.WARNING)]) is False
    assert condition.matches([_pii_finding(Severity.HIGH)]) is True
    assert condition.matches([_pii_finding(Severity.CRITICAL)]) is True


def test_condition_metadata_below_gate():

    condition = PolicyCondition(
        guardrail_name="hallucination_detector",
        metadata_key="groundedness_score",
        metadata_below=0.30
    )

    assert condition.matches([_hallucination_finding(0.50)]) is False
    assert condition.matches([_hallucination_finding(0.10)]) is True


def test_rule_requires_all_conditions_to_match():

    rule = PolicyRule(
        name="test_rule",
        conditions=[
            PolicyCondition(guardrail_name="pii_guard", min_severity=Severity.HIGH),
            PolicyCondition(
                guardrail_name="hallucination_detector",
                metadata_key="groundedness_score",
                metadata_below=0.30
            )
        ],
        action=Action.BLOCK
    )

    only_pii = [_pii_finding(Severity.CRITICAL)]
    both = [_pii_finding(Severity.CRITICAL), _hallucination_finding(0.10)]

    assert rule.matches(only_pii) is False
    assert rule.matches(both) is True


def test_engine_returns_default_action_when_no_rule_matches():

    engine = PolicyEngine(rules=[])

    action = engine.evaluate([_pii_finding(Severity.INFO)], default_action=Action.ALLOW)

    assert action == Action.ALLOW


def test_engine_never_downgrades_the_default_action():

    warn_only_rule = PolicyRule(
        name="always_warn",
        conditions=[PolicyCondition(guardrail_name="pii_guard", triggered=True)],
        action=Action.WARN
    )
    engine = PolicyEngine(rules=[warn_only_rule])

    action = engine.evaluate([_pii_finding(Severity.CRITICAL)], default_action=Action.BLOCK)

    assert action == Action.BLOCK


def test_default_policies_pii_and_high_blocks():

    engine = PolicyEngine.default_policies()

    action = engine.evaluate(
        [_pii_finding(Severity.HIGH)],
        default_action=Action.REDACT
    )

    assert action == Action.BLOCK


def test_default_policies_pii_below_high_does_not_block():

    engine = PolicyEngine.default_policies()

    action = engine.evaluate(
        [_pii_finding(Severity.WARNING)],
        default_action=Action.REDACT
    )

    assert action == Action.REDACT


def test_default_policies_hallucination_below_030_warns():

    engine = PolicyEngine.default_policies()

    action = engine.evaluate(
        [_hallucination_finding(0.10)],
        default_action=Action.ALLOW
    )

    assert action == Action.WARN


def test_default_policies_hallucination_above_030_does_not_warn():

    engine = PolicyEngine.default_policies()

    action = engine.evaluate(
        [_hallucination_finding(0.50)],
        default_action=Action.ALLOW
    )

    assert action == Action.ALLOW
