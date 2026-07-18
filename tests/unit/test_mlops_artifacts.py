import pytest

from mlops.artifacts import ArtifactNotFoundError
from mlops.artifacts import ArtifactRegistry


def test_save_creates_version_one():

    registry = ArtifactRegistry()

    version = registry.save("prompt-1", "prompt_template", {"text": "Answer using context."})

    assert version.version == 1


def test_repeated_saves_are_immutable_and_append_only():

    registry = ArtifactRegistry()
    registry.save("prompt-1", "prompt_template", {"text": "v1"})
    registry.save("prompt-1", "prompt_template", {"text": "v2"})
    third = registry.save("prompt-1", "prompt_template", {"text": "v3"})

    history = registry.history("prompt-1")

    assert [v.version for v in history] == [1, 2, 3]
    assert history[0].content == {"text": "v1"}
    assert third.version == 3
    assert third.content == {"text": "v3"}


def test_get_latest_returns_most_recent_version():

    registry = ArtifactRegistry()
    registry.save("chunking", "chunking_config", {"chunk_size": 900})
    registry.save("chunking", "chunking_config", {"chunk_size": 512})

    latest = registry.get_latest("chunking")

    assert latest.version == 2
    assert latest.content == {"chunk_size": 512}


def test_get_specific_version():

    registry = ArtifactRegistry()
    registry.save("chunking", "chunking_config", {"chunk_size": 900})
    registry.save("chunking", "chunking_config", {"chunk_size": 512})

    first = registry.get_version("chunking", 1)

    assert first.content == {"chunk_size": 900}


def test_missing_artifact_raises():

    registry = ArtifactRegistry()

    with pytest.raises(ArtifactNotFoundError):
        registry.get_latest("does-not-exist")

    with pytest.raises(ArtifactNotFoundError):
        registry.history("does-not-exist")


def test_missing_version_raises():

    registry = ArtifactRegistry()
    registry.save("chunking", "chunking_config", {"chunk_size": 900})

    with pytest.raises(ArtifactNotFoundError):
        registry.get_version("chunking", 5)


def test_list_artifact_ids():

    registry = ArtifactRegistry()
    registry.save("a", "prompt_template", {})
    registry.save("b", "policy", {})

    assert set(registry.list_artifact_ids()) == {"a", "b"}


def test_export_import_state_round_trips():

    registry = ArtifactRegistry()
    registry.save("a", "prompt_template", {"text": "hello"}, created_by="alice")
    registry.save("a", "prompt_template", {"text": "hello v2"})

    state = registry.export_state()
    restored = ArtifactRegistry()
    restored.import_state(state)

    assert restored.history("a") == registry.history("a")
