import re
from typing import Any

from rag.vector_store.opensearch_store import build_knn_index_mapping

VERSIONED_INDEX_PATTERN = re.compile(r"^(?P<base>.+)-v(?P<version>\d+)$")


class AliasNotFoundError(RuntimeError):
    pass


class OpenSearchIndexManager:
    """
    Versioned-index + alias workflow: rag-v1, rag-v2, ... with an alias
    (e.g. rag-prod) pointing at exactly one of them at a time. Changing the
    embedding model, chunking parameters, retrieval settings, or index
    mapping means creating a new version rather than mutating the
    production index in place - ingest and evaluate the new version, then
    atomically repoint the alias. Rollback is just switching the alias
    back to a still-present older version; nothing is deleted by this
    class unless delete_version() is called explicitly.
    """

    def __init__(
        self,
        client: Any,
        base_index_name: str
    ) -> None:
        self.client = client
        self.base_index_name = base_index_name

    def index_name_for_version(
        self,
        version: int
    ) -> str:
        return f"{self.base_index_name}-v{version}"

    def create_version(
        self,
        version: int,
        embedding_dimensions: int
    ) -> str:
        """Creates {base}-v{version} with the standard chunk mapping if it
        doesn't already exist. Returns the index name either way."""
        index_name = self.index_name_for_version(version)

        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(
                index=index_name,
                body=build_knn_index_mapping(embedding_dimensions)
            )

        return index_name

    def list_versions(self) -> list[int]:
        names = self.client.indices.list_names(f"{self.base_index_name}-v*")
        versions = []

        for name in names:
            match = VERSIONED_INDEX_PATTERN.match(name)

            if match and match.group("base") == self.base_index_name:
                versions.append(int(match.group("version")))

        return sorted(versions)

    def next_version(self) -> int:
        existing = self.list_versions()
        return (max(existing) + 1) if existing else 1

    def current_index(
        self,
        alias_name: str
    ) -> str | None:
        """The concrete index an alias currently points to, or None if the
        alias doesn't exist yet (e.g. before the first switch_alias call)."""
        aliases = self.client.indices.get_alias(alias_name)

        if not aliases:
            return None

        return next(iter(aliases.keys()))

    def switch_alias(
        self,
        alias_name: str,
        to_index: str
    ) -> str:
        """
        Atomically repoints alias_name at to_index - removes it from
        whatever index it currently points to (if any) and adds it to
        to_index in a single OpenSearch request, so queries against the
        alias never see it missing or resolving to two indexes at once.
        """
        current = self.current_index(alias_name)
        actions: list[dict[str, Any]] = []

        if current and current != to_index:
            actions.append({"remove": {"index": current, "alias": alias_name}})

        actions.append({"add": {"index": to_index, "alias": alias_name}})
        self.client.indices.update_aliases(actions)
        return to_index

    def rollback(
        self,
        alias_name: str,
        to_version: int
    ) -> str:
        """Same mechanism as switch_alias, named for the rollback use case -
        repoints the alias to an older, still-existing version."""
        target_index = self.index_name_for_version(to_version)

        if not self.client.indices.exists(index=target_index):
            raise AliasNotFoundError(
                f"cannot roll back to {target_index!r} - that index no longer exists"
            )

        return self.switch_alias(alias_name, target_index)

    def delete_version(
        self,
        version: int
    ) -> None:
        """
        Deletes a specific versioned index outright - never called
        automatically by this class. A version currently pointed at by an
        alias should be repointed elsewhere first; this method doesn't
        check that for you, since "is this alias's current target" is a
        judgment call the caller should make deliberately, not a rule this
        class silently enforces.
        """
        self.client.indices.delete(self.index_name_for_version(version), ignore=[404])
