"""
Live verification for the pieces that were only unit-tested with a fake
client, never actually run against a real OpenSearch domain:
  - OpenSearchIndexManager (versioned indexes + alias switching + rollback)
  - full chunk lineage/ACL field round-trip (document_version, tenant_id,
    access_groups, classification) through the real mapping
  - RAGService._filter_by_access() against an OpenSearch-backed store,
    not just InMemoryVectorStore (every earlier RAGService test used the
    in-memory store)

Not part of the pytest suite - run directly, same as opensearch_smoke_test.py.
"""
import sys

from app.config import load_settings
from rag.chunking.chunk import Chunk
from rag.vector_store.opensearch_client_factory import build_opensearch_client
from rag.vector_store.opensearch_index_manager import OpenSearchIndexManager
from rag.vector_store.opensearch_store import OpenSearchVectorStore

BASE_INDEX = "verify-phase2"
ALIAS = "verify-phase2-alias"
DIMENSIONS = 4


def verify_index_manager(client) -> bool:
    print("\n--- OpenSearchIndexManager (versioned indexes + alias) ---")
    manager = OpenSearchIndexManager(client=client, base_index_name=BASE_INDEX)

    for v in manager.list_versions():
        manager.delete_version(v)
    print("cleaned up any pre-existing versions")

    v1 = manager.create_version(1, DIMENSIONS)
    print(f"created {v1}")
    v2 = manager.create_version(2, DIMENSIONS)
    print(f"created {v2}")

    versions = manager.list_versions()
    print(f"list_versions() -> {versions}")
    if versions != [1, 2]:
        print(f"FAILED: expected [1, 2], got {versions}")
        return False

    next_v = manager.next_version()
    print(f"next_version() -> {next_v}")
    if next_v != 3:
        print(f"FAILED: expected 3, got {next_v}")
        return False

    manager.switch_alias(ALIAS, v1)
    current = manager.current_index(ALIAS)
    print(f"switched alias to {v1}, current_index() -> {current}")
    if current != v1:
        print(f"FAILED: expected {v1}, got {current}")
        return False

    manager.switch_alias(ALIAS, v2)
    current = manager.current_index(ALIAS)
    print(f"switched alias to {v2}, current_index() -> {current}")
    if current != v2:
        print(f"FAILED: expected {v2}, got {current}")
        return False

    manager.rollback(ALIAS, to_version=1)
    current = manager.current_index(ALIAS)
    print(f"rolled back, current_index() -> {current}")
    if current != v1:
        print(f"FAILED: expected {v1} after rollback, got {current}")
        return False

    manager.delete_version(1)
    manager.delete_version(2)
    print("OK: create/list/next/switch/rollback/delete all verified live")
    return True


def verify_lineage_and_acl_round_trip(client) -> bool:
    print("\n--- chunk lineage + ACL field round-trip ---")
    index_name = "verify-phase2-lineage"
    store = OpenSearchVectorStore(client=client, index_name=index_name, embedding_dimensions=DIMENSIONS)
    store.ensure_index(DIMENSIONS)

    chunk = Chunk(
        chunk_id="lineage:0",
        document_id="lineage-doc",
        document_version=2,
        chunk_index=0,
        text="Confidential quarterly revenue figures for the finance team.",
        source="finance.md",
        document_type="markdown",
        content_hash="abc123",
        chunking_version="recursive:900:120:80",
        embedding_provider="hashing",
        embedding_model="hashing-4",
        embedding_version="4",
        tenant_id="acme-corp",
        access_groups=["finance", "executives"],
        classification="confidential"
    )
    store.add(chunk, [1.0, 0.0, 0.0, 0.0])

    import time
    time.sleep(1)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    if not results:
        print("FAILED: no results returned")
        return False

    retrieved = results[0].chunk
    checks = {
        "document_version": (retrieved.document_version, 2),
        "content_hash": (retrieved.content_hash, "abc123"),
        "chunking_version": (retrieved.chunking_version, "recursive:900:120:80"),
        "embedding_provider": (retrieved.embedding_provider, "hashing"),
        "tenant_id": (retrieved.tenant_id, "acme-corp"),
        "access_groups": (retrieved.access_groups, ["finance", "executives"]),
        "classification": (retrieved.classification, "confidential"),
    }
    all_ok = True
    for field, (actual, expected) in checks.items():
        status = "OK" if actual == expected else "FAILED"
        if actual != expected:
            all_ok = False
        print(f"{status}: {field} = {actual!r} (expected {expected!r})")

    store.delete_by_document("lineage-doc")
    client.indices.delete(index_name, ignore=[404])
    return all_ok


def verify_access_control_against_opensearch(client) -> bool:
    print("\n--- RAGService document-level authorization against OpenSearch ---")
    from app.services.rag_service import RAGService

    index_name = "verify-phase2-acl"
    store = OpenSearchVectorStore(client=client, index_name=index_name, embedding_dimensions=384)
    store.ensure_index(384)

    service = RAGService(vector_store=store)
    restricted = Chunk(
        chunk_id="acl:0", document_id="acl-doc", chunk_index=0,
        text="Confidential salary information for engineering staff.",
        source="hr.md", document_type="markdown", access_groups=["hr"]
    )
    store.add(restricted, service.embedder.embed(restricted.text))

    import time
    time.sleep(1)

    denied = service.ask("salary information", access_groups=["engineering"])
    allowed = service.ask("salary information", access_groups=["hr"])

    client.indices.delete(index_name, ignore=[404])

    denied_ok = denied.sources == []
    allowed_ok = bool(allowed.sources)
    print(f"{'OK' if denied_ok else 'FAILED'}: unauthorized caller got 0 sources ({len(denied.sources)})")
    print(f"{'OK' if allowed_ok else 'FAILED'}: authorized caller got sources ({len(allowed.sources)})")
    return denied_ok and allowed_ok


def main() -> int:
    settings = load_settings()

    if not settings.opensearch_host:
        print("FAILED: OPENSEARCH_HOST is not set")
        return 1

    client = build_opensearch_client(
        host=settings.opensearch_host,
        region=settings.aws_region,
        port=settings.opensearch_port,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        connect_timeout=settings.opensearch_connect_timeout,
        max_retries=settings.opensearch_max_retries
    )

    results = [
        verify_index_manager(client),
        verify_lineage_and_acl_round_trip(client),
        verify_access_control_against_opensearch(client),
    ]

    if all(results):
        print("\nALL LIVE VERIFICATIONS PASSED")
        return 0

    print("\nSOME LIVE VERIFICATIONS FAILED - see above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
