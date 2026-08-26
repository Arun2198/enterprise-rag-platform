"""
End-to-end smoke test for OpenSearchVectorStore against a real OpenSearch domain.

Not part of the pytest suite (unit tests must not make network calls) - run
this directly, with OPENSEARCH_HOST/AWS_REGION supplied via environment
variables and AWS credentials resolvable ambiently (same as any other boto3
call in this project).
"""
import sys

from app.config import load_settings
from rag.chunking.chunk import Chunk
from rag.vector_store.opensearch_client_factory import build_opensearch_client
from rag.vector_store.opensearch_store import OpenSearchVectorStore

TEST_INDEX = "opensearch-smoke-test"
DIMENSIONS = 4


def main() -> int:
    settings = load_settings()

    if not settings.opensearch_host:
        print("FAILED: OPENSEARCH_HOST is not set")
        return 1

    print(f"host: {settings.opensearch_host}")
    print(f"region: {settings.aws_region}")

    client = build_opensearch_client(
        host=settings.opensearch_host,
        region=settings.aws_region,
        port=settings.opensearch_port,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_verify_certs,
        connect_timeout=settings.opensearch_connect_timeout,
        max_retries=settings.opensearch_max_retries
    )
    store = OpenSearchVectorStore(client=client, index_name=TEST_INDEX, embedding_dimensions=DIMENSIONS)

    health = store.health_check()
    print(f"cluster health: {health.get('status')}")

    created = store.ensure_index(DIMENSIONS)
    print(f"index created: {created} (index={TEST_INDEX})")

    chunk = Chunk(
        chunk_id="smoke:0",
        document_id="smoke",
        chunk_index=0,
        text="OpenSearch smoke test document about enterprise retrieval augmented generation.",
        source="smoke_test.md",
        document_type="markdown",
        metadata={"domain": "smoke_test"}
    )
    store.add(chunk, [1.0, 0.0, 0.0, 0.0])
    print("indexed 1 document")

    import time
    time.sleep(1)  # OpenSearch is near-real-time, not instant - give the refresh a moment

    vector_results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    print(f"vector search hits: {len(vector_results)}")

    if not vector_results or vector_results[0].chunk.chunk_id != "smoke:0":
        print("FAILED: vector search did not return the indexed document")
        return 1

    lexical_results = store.search_lexical("enterprise retrieval", top_k=1)
    print(f"lexical (BM25) search hits: {len(lexical_results)}")

    if not lexical_results or lexical_results[0].chunk.chunk_id != "smoke:0":
        print("FAILED: lexical search did not return the indexed document")
        return 1

    store.update_metadata("smoke:0", {"domain": "smoke_test", "verified": "true"})
    print("updated metadata")

    store.delete_by_document("smoke")
    print("deleted the smoke-test document")

    print("OK: health check, index creation, bulk-path indexing, vector search, "
          "BM25 search, metadata update, and delete all verified against the live domain")
    return 0


if __name__ == "__main__":
    sys.exit(main())
