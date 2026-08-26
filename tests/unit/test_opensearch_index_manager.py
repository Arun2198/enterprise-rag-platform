import pytest

from rag.vector_store.opensearch_index_manager import AliasNotFoundError
from rag.vector_store.opensearch_index_manager import OpenSearchIndexManager


class FakeIndicesClient:

    def __init__(self):
        self.existing_indexes: set[str] = set()
        self.created_indexes: list[dict] = []
        self.deleted_indexes: list[str] = []
        self.alias_targets: dict[str, str] = {}
        self.update_aliases_calls: list[list[dict]] = []

    def exists(self, index):
        return index in self.existing_indexes

    def create(self, index, body):
        self.existing_indexes.add(index)
        self.created_indexes.append({"index": index, "body": body})

    def delete(self, index, ignore=None):
        self.existing_indexes.discard(index)
        self.deleted_indexes.append(index)

    def get_alias(self, name):
        target = self.alias_targets.get(name)
        return {target: {"aliases": {name: {}}}} if target else {}

    def update_aliases(self, actions):
        self.update_aliases_calls.append(actions)

        for action in actions:
            if "remove" in action:
                alias = action["remove"]["alias"]
                self.alias_targets.pop(alias, None)
            if "add" in action:
                self.alias_targets[action["add"]["alias"]] = action["add"]["index"]

    def list_names(self, pattern):
        base = pattern.rstrip("*")
        return sorted(name for name in self.existing_indexes if name.startswith(base))


class FakeClient:

    def __init__(self):
        self.indices = FakeIndicesClient()


def test_create_version_creates_the_versioned_index_with_the_knn_mapping():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    index_name = manager.create_version(1, embedding_dimensions=384)

    assert index_name == "rag-v1"
    assert client.indices.existing_indexes == {"rag-v1"}
    mapping = client.indices.created_indexes[0]["body"]
    assert mapping["mappings"]["properties"]["embedding"]["dimension"] == 384


def test_create_version_is_idempotent():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    manager.create_version(1, embedding_dimensions=384)
    manager.create_version(1, embedding_dimensions=384)

    assert len(client.indices.created_indexes) == 1


def test_list_versions_returns_sorted_version_numbers():

    client = FakeClient()
    client.indices.existing_indexes = {"rag-v3", "rag-v1", "rag-v2", "other-v1"}
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    assert manager.list_versions() == [1, 2, 3]


def test_next_version_is_one_when_nothing_exists_yet():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    assert manager.next_version() == 1


def test_next_version_increments_past_the_highest_existing_version():

    client = FakeClient()
    client.indices.existing_indexes = {"rag-v1", "rag-v2"}
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    assert manager.next_version() == 3


def test_current_index_is_none_before_any_switch():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    assert manager.current_index("rag-prod") is None


def test_switch_alias_points_the_alias_at_the_new_index():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")
    manager.create_version(1, embedding_dimensions=384)

    manager.switch_alias("rag-prod", "rag-v1")

    assert manager.current_index("rag-prod") == "rag-v1"


def test_switch_alias_atomically_moves_from_the_old_to_the_new_index():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")
    manager.create_version(1, embedding_dimensions=384)
    manager.create_version(2, embedding_dimensions=384)
    manager.switch_alias("rag-prod", "rag-v1")

    manager.switch_alias("rag-prod", "rag-v2")

    assert manager.current_index("rag-prod") == "rag-v2"
    last_actions = client.indices.update_aliases_calls[-1]
    assert {"remove": {"index": "rag-v1", "alias": "rag-prod"}} in last_actions
    assert {"add": {"index": "rag-v2", "alias": "rag-prod"}} in last_actions


def test_rollback_repoints_the_alias_to_an_older_version():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")
    manager.create_version(1, embedding_dimensions=384)
    manager.create_version(2, embedding_dimensions=384)
    manager.switch_alias("rag-prod", "rag-v2")

    manager.rollback("rag-prod", to_version=1)

    assert manager.current_index("rag-prod") == "rag-v1"


def test_rollback_to_a_version_that_was_never_created_raises():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")

    with pytest.raises(AliasNotFoundError):
        manager.rollback("rag-prod", to_version=99)


def test_delete_version_removes_the_index():

    client = FakeClient()
    manager = OpenSearchIndexManager(client=client, base_index_name="rag")
    manager.create_version(1, embedding_dimensions=384)

    manager.delete_version(1)

    assert "rag-v1" not in client.indices.existing_indexes
