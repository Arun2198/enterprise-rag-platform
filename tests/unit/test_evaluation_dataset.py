import json

import pytest

from evaluation.dataset import DatasetValidationError
from evaluation.dataset import load_dataset


def _write_dataset(tmp_path, payload):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _minimal_payload(**overrides):
    payload = {
        "name": "test-dataset",
        "queries": [
            {
                "id": "Q001",
                "query": "What is the leave policy?",
                "relevant_chunk_ids": ["doc:0"],
                "category": "HR",
                "difficulty": "easy"
            }
        ]
    }
    payload.update(overrides)
    return payload


def test_load_valid_dataset(tmp_path):

    path = _write_dataset(tmp_path, _minimal_payload())

    dataset = load_dataset(path)

    assert dataset.name == "test-dataset"
    assert len(dataset.queries) == 1
    assert dataset.queries[0].id == "Q001"
    assert dataset.queries[0].category == "HR"
    assert dataset.queries[0].difficulty == "easy"


def test_load_dataset_supports_multiple_relevant_chunks(tmp_path):

    payload = _minimal_payload()
    payload["queries"][0]["relevant_chunk_ids"] = ["doc:0", "doc:1", "doc:2"]
    path = _write_dataset(tmp_path, payload)

    dataset = load_dataset(path)

    assert dataset.queries[0].relevant_chunk_ids == ["doc:0", "doc:1", "doc:2"]


def test_missing_file_raises_validation_error():

    with pytest.raises(DatasetValidationError):
        load_dataset("does/not/exist.json")


def test_invalid_json_raises_validation_error(tmp_path):

    path = tmp_path / "broken.json"
    path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_dataset(str(path))


def test_empty_queries_list_is_rejected(tmp_path):

    path = _write_dataset(tmp_path, _minimal_payload(queries=[]))

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_missing_name_is_rejected(tmp_path):

    payload = _minimal_payload()
    del payload["name"]
    path = _write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_query_missing_relevant_chunk_ids_is_rejected(tmp_path):

    payload = _minimal_payload()
    del payload["queries"][0]["relevant_chunk_ids"]
    path = _write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_query_with_empty_relevant_chunk_ids_is_rejected(tmp_path):

    payload = _minimal_payload()
    payload["queries"][0]["relevant_chunk_ids"] = []
    path = _write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_invalid_difficulty_is_rejected(tmp_path):

    payload = _minimal_payload()
    payload["queries"][0]["difficulty"] = "impossible"
    path = _write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_duplicate_query_ids_are_rejected(tmp_path):

    payload = _minimal_payload()
    payload["queries"].append(dict(payload["queries"][0]))
    path = _write_dataset(tmp_path, payload)

    with pytest.raises(DatasetValidationError):
        load_dataset(path)


def test_category_and_difficulty_are_optional(tmp_path):

    payload = _minimal_payload()
    del payload["queries"][0]["category"]
    del payload["queries"][0]["difficulty"]
    path = _write_dataset(tmp_path, payload)

    dataset = load_dataset(path)

    assert dataset.queries[0].category is None
    assert dataset.queries[0].difficulty is None


def test_real_golden_dataset_loads_and_validates():

    dataset = load_dataset("evaluation/golden_dataset.json")

    assert dataset.name == "ai-rmf-1stdraft"
    assert len(dataset.queries) == 20
    assert dataset.source_documents == ["sample_documents/AI-RMF-1stdraft.pdf"]
