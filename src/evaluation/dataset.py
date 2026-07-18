import json
from pathlib import Path

from evaluation.schemas import GoldenDataset
from evaluation.schemas import GoldenQuery

VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class DatasetValidationError(ValueError):
    pass


def load_dataset(
    path: str
) -> GoldenDataset:
    file_path = Path(path)

    if not file_path.exists():
        raise DatasetValidationError(f"dataset not found: {path}")

    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as ex:
        raise DatasetValidationError(f"{path}: invalid JSON - {ex}") from ex

    if not isinstance(raw, dict):
        raise DatasetValidationError(f"{path}: dataset must be a JSON object")

    name = raw.get("name")

    if not name or not isinstance(name, str):
        raise DatasetValidationError(f"{path}: missing required field 'name'")

    raw_queries = raw.get("queries")

    if not isinstance(raw_queries, list) or not raw_queries:
        raise DatasetValidationError(
            f"{path}: 'queries' must be a non-empty list"
        )

    queries = [
        _parse_query(path, index, entry)
        for index, entry in enumerate(raw_queries)
    ]
    _check_duplicate_ids(path, queries)

    return GoldenDataset(
        name=name,
        description=raw.get("description"),
        source_documents=raw.get("source_documents", []),
        queries=queries
    )


def _parse_query(
    path: str,
    index: int,
    entry: object
) -> GoldenQuery:
    if not isinstance(entry, dict):
        raise DatasetValidationError(
            f"{path}: queries[{index}] must be an object"
        )

    query_id = entry.get("id")

    if not query_id or not isinstance(query_id, str):
        raise DatasetValidationError(
            f"{path}: queries[{index}] missing required field 'id'"
        )

    query_text = entry.get("query")

    if not query_text or not isinstance(query_text, str):
        raise DatasetValidationError(
            f"{path}: queries[{index}] ('{query_id}') missing required field 'query'"
        )

    relevant_chunk_ids = entry.get("relevant_chunk_ids")

    if not isinstance(relevant_chunk_ids, list) or not relevant_chunk_ids:
        raise DatasetValidationError(
            f"{path}: queries[{index}] ('{query_id}') 'relevant_chunk_ids' "
            "must be a non-empty list"
        )

    if not all(isinstance(chunk_id, str) for chunk_id in relevant_chunk_ids):
        raise DatasetValidationError(
            f"{path}: queries[{index}] ('{query_id}') 'relevant_chunk_ids' "
            "must all be strings"
        )

    difficulty = entry.get("difficulty")

    if difficulty is not None and difficulty not in VALID_DIFFICULTIES:
        raise DatasetValidationError(
            f"{path}: queries[{index}] ('{query_id}') difficulty '{difficulty}' "
            f"must be one of {sorted(VALID_DIFFICULTIES)}"
        )

    return GoldenQuery(
        id=query_id,
        query=query_text,
        relevant_chunk_ids=relevant_chunk_ids,
        category=entry.get("category"),
        difficulty=difficulty
    )


def _check_duplicate_ids(
    path: str,
    queries: list[GoldenQuery]
) -> None:
    seen: set[str] = set()

    for query in queries:
        if query.id in seen:
            raise DatasetValidationError(
                f"{path}: duplicate query id '{query.id}'"
            )

        seen.add(query.id)
