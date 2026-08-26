import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

VALID_CATEGORIES = {"unanswerable", "adversarial"}


class RobustnessDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class RobustnessCase:
    """
    Neither "unanswerable" nor "adversarial" queries fit evaluation's Layer
    1 recall/precision schema (GoldenQuery.relevant_chunk_ids must be a
    non-empty list - there's structurally no "correct chunk" for a query
    the corpus doesn't answer, and an adversarial query is testing guardrail
    behavior, not retrieval quality). This is a parallel, smaller dataset
    that checks system *behavior* against a real RAGService instead of
    retrieval recall against golden chunk ids.
    """
    id: str
    query: str
    category: str
    expect_abstention: bool = False
    expect_block: bool = False
    description: str | None = None


@dataclass(frozen=True)
class RobustnessDataset:
    name: str
    cases: list[RobustnessCase]
    description: str | None = None


@dataclass(frozen=True)
class RobustnessResult:
    case_id: str
    query: str
    category: str
    passed: bool
    answer: str
    observed_action: str
    detail: str


@dataclass(frozen=True)
class RobustnessReport:
    dataset_name: str
    results: list[RobustnessResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0


class AsksQuestions(Protocol):
    """
    The one method run_robustness_eval actually needs - satisfied by the
    real RAGService, so this stays testable without importing app-layer
    code into the standalone evaluation package.
    """

    def ask(self, query: str, top_k: int = 5) -> object:
        ...


def load_robustness_dataset(
    path: str
) -> RobustnessDataset:
    file_path = Path(path)

    if not file_path.exists():
        raise RobustnessDatasetError(f"dataset not found: {path}")

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise RobustnessDatasetError(f"{path}: invalid JSON - {ex}") from ex

    name = raw.get("name")

    if not name or not isinstance(name, str):
        raise RobustnessDatasetError(f"{path}: missing required field 'name'")

    raw_cases = raw.get("cases")

    if not isinstance(raw_cases, list) or not raw_cases:
        raise RobustnessDatasetError(f"{path}: 'cases' must be a non-empty list")

    cases = [_parse_case(path, index, entry) for index, entry in enumerate(raw_cases)]
    seen_ids: set[str] = set()

    for case in cases:
        if case.id in seen_ids:
            raise RobustnessDatasetError(f"{path}: duplicate case id '{case.id}'")

        seen_ids.add(case.id)

    return RobustnessDataset(name=name, description=raw.get("description"), cases=cases)


def _parse_case(
    path: str,
    index: int,
    entry: object
) -> RobustnessCase:
    if not isinstance(entry, dict):
        raise RobustnessDatasetError(f"{path}: cases[{index}] must be an object")

    case_id = entry.get("id")

    if not case_id or not isinstance(case_id, str):
        raise RobustnessDatasetError(f"{path}: cases[{index}] missing required field 'id'")

    query = entry.get("query")

    if not query or not isinstance(query, str):
        raise RobustnessDatasetError(f"{path}: cases[{index}] ('{case_id}') missing required field 'query'")

    category = entry.get("category")

    if category not in VALID_CATEGORIES:
        raise RobustnessDatasetError(
            f"{path}: cases[{index}] ('{case_id}') category '{category}' "
            f"must be one of {sorted(VALID_CATEGORIES)}"
        )

    return RobustnessCase(
        id=case_id,
        query=query,
        category=category,
        expect_abstention=bool(entry.get("expect_abstention", False)),
        expect_block=bool(entry.get("expect_block", False)),
        description=entry.get("description")
    )


def run_robustness_eval(
    rag_service: AsksQuestions,
    dataset: RobustnessDataset,
    abstention_message: str
) -> RobustnessReport:
    results = []

    for case in dataset.cases:
        response = rag_service.ask(case.query)
        # RAGService.ask() only ever returns sources=[] and confidence=0.0
        # together when a guardrail's BLOCK action fired (see
        # app.services.rag_service.RAGService.ask) - a reliable behavioral
        # signal without reaching into guardrail_flags internals.
        blocked = not response.sources and response.confidence == 0.0
        abstained = response.answer == abstention_message
        observed_action = "block" if blocked else ("abstain" if abstained else "answered")

        passed = True
        detail = "ok"

        if case.expect_block and not blocked:
            passed = False
            detail = f"expected a guardrail block, got '{observed_action}'"
        elif case.expect_abstention and not (abstained or blocked):
            passed = False
            detail = f"expected abstention, got '{observed_action}' with a confident answer"

        results.append(RobustnessResult(
            case_id=case.id,
            query=case.query,
            category=case.category,
            passed=passed,
            answer=response.answer,
            observed_action=observed_action,
            detail=detail
        ))

    return RobustnessReport(dataset_name=dataset.name, results=results)
