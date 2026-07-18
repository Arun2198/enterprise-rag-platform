from dataclasses import dataclass

from rag.guardrails.base import ACTION_RANK
from rag.guardrails.base import SEVERITY_RANK
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import Severity


@dataclass(frozen=True)
class PolicyCondition:
    """
    A single condition checked against one guardrail's finding. Every
    non-None field must hold for the condition to match.
    """
    guardrail_name: str
    triggered: bool = True
    min_severity: Severity | None = None
    metadata_key: str | None = None
    metadata_below: float | None = None

    def matches(
        self,
        findings: list[GuardrailFinding]
    ) -> bool:
        candidates = [
            finding
            for finding in findings
            if finding.guardrail_name == self.guardrail_name
        ]

        if not candidates:
            return False

        finding = candidates[0]

        if self.triggered and not finding.triggered:
            return False

        if self.min_severity is not None:
            if SEVERITY_RANK[finding.severity] < SEVERITY_RANK[self.min_severity]:
                return False

        if self.metadata_key is not None and self.metadata_below is not None:
            value = finding.metadata.get(self.metadata_key)

            if value is None or not (value < self.metadata_below):
                return False

        return True


@dataclass(frozen=True)
class PolicyRule:
    name: str
    conditions: list[PolicyCondition]
    action: Action

    def matches(
        self,
        findings: list[GuardrailFinding]
    ) -> bool:
        return all(condition.matches(findings) for condition in self.conditions)


class PolicyEngine:
    """
    Evaluates configurable IF-THEN rules over a batch of guardrail
    findings and can escalate (never downgrade) the action a
    GuardrailManager would otherwise take. Rules are plain data - build
    the list from config/JSON in production instead of Python if needed.
    Not attached by GuardrailManager.default() - Phase 1 findings apply
    their own suggested action directly; wire a PolicyEngine in explicitly
    to opt into this.
    """

    def __init__(
        self,
        rules: list[PolicyRule] | None = None
    ) -> None:
        self.rules = rules or []

    def evaluate(
        self,
        findings: list[GuardrailFinding],
        default_action: Action
    ) -> Action:
        matched = [rule.action for rule in self.rules if rule.matches(findings)]

        if not matched:
            return default_action

        return max([default_action, *matched], key=lambda action: ACTION_RANK[action])

    @classmethod
    def default_policies(cls) -> "PolicyEngine":
        return cls(
            rules=[
                PolicyRule(
                    name="pii_high_severity_block",
                    conditions=[
                        PolicyCondition(
                            guardrail_name="pii_guard",
                            min_severity=Severity.HIGH
                        )
                    ],
                    action=Action.BLOCK
                ),
                PolicyRule(
                    name="hallucination_low_groundedness_warn",
                    conditions=[
                        PolicyCondition(
                            guardrail_name="hallucination_detector",
                            metadata_key="groundedness_score",
                            metadata_below=0.30
                        )
                    ],
                    action=Action.WARN
                )
            ]
        )
