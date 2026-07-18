from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Protocol

from rag.retrieval.hybrid_retrieval import RetrievedChunk


class GuardrailStage(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    WARN = "warn"
    BLOCK = "block"
    ESCALATE = "escalate"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}

ACTION_RANK: dict[Action, int] = {
    Action.ALLOW: 0,
    Action.WARN: 1,
    Action.REDACT: 2,
    Action.ESCALATE: 3,
    Action.BLOCK: 4,
}


@dataclass(frozen=True)
class GuardrailContext:
    """
    Everything a guardrail might need. Input-stage guardrails only look at
    `query`; output-stage guardrails also get `answer` and the chunks it
    was generated from.
    """
    query: str
    answer: str | None = None
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass(frozen=True)
class GuardrailFinding:
    guardrail_name: str
    triggered: bool
    severity: Severity
    action: Action
    message: str
    redacted_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Guardrail(Protocol):
    """
    Every guardrail - MVP or enterprise, regex-based or model-backed -
    implements this same shape. New ones register with a GuardrailManager;
    nothing else in the pipeline needs to change.
    """
    name: str
    stage: GuardrailStage

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        ...
