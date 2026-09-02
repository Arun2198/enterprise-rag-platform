from typing import Any

from rag.chunking.chunk import Chunk
from rag.vector_store.in_memory_store import SearchResult

DEFAULT_SPACE_TYPE = "cosinesimil"


def build_knn_index_mapping(dimensions: int) -> dict[str, Any]:
    """
    Shared by OpenSearchVectorStore.ensure_index() and
    OpenSearchIndexManager.create_version() so a versioned index and an
    ad-hoc one are never accidentally mapped differently.
    """
    return {
        "settings": {
            "index": {"knn": True}
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "document_id": {"type": "keyword"},
                "document_version": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "document_type": {"type": "keyword"},
                "owner": {"type": "keyword"},
                "parent_section": {"type": "text"},
                "content_hash": {"type": "keyword"},
                "chunking_version": {"type": "keyword"},
                "embedding_provider": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding_version": {"type": "keyword"},
                "tenant_id": {"type": "keyword"},
                "access_groups": {"type": "keyword"},
                "classification": {"type": "keyword"},
                "created_at": {"type": "date"},
                "indexed_at": {"type": "date"},
                "metadata": {"type": "object", "enabled": True},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": DEFAULT_SPACE_TYPE,
                        "engine": "lucene"
                    }
                }
            }
        }
    }


class EmbeddingDimensionError(ValueError):
    pass


class BulkIndexError(RuntimeError):

    def __init__(self, failed_items: list[dict[str, Any]]) -> None:
        self.failed_items = failed_items
        super().__init__(
            f"{len(failed_items)} item(s) failed during bulk indexing: "
            f"{failed_items[:3]}{'...' if len(failed_items) > 3 else ''}"
        )


class OpenSearchVectorStore:
    """
    OpenSearch adapter - production vector + lexical store.

    The client is injected (built via opensearch_client_factory.build_opensearch_client
    for real deployments) to keep this class itself free of AWS-auth concerns and
    trivially testable with a fake client.
    """

    def __init__(
        self,
        client: Any,
        index_name: str,
        embedding_dimensions: int | None = None
    ) -> None:
        self.client = client
        self.index_name = index_name
        self.embedding_dimensions = embedding_dimensions

    def ensure_index(
        self,
        embedding_dimensions: int | None = None
    ) -> bool:
        """
        Creates the index with a k-NN-enabled mapping if it doesn't already
        exist. Returns True if the index was created, False if it already
        existed. Safe to call on every startup - a no-op once the index is
        there.
        """
        dimensions = embedding_dimensions or self.embedding_dimensions

        if dimensions is None:
            raise EmbeddingDimensionError(
                "embedding_dimensions must be known to create the index "
                "mapping - pass it to ensure_index() or the constructor."
            )

        if self.client.indices.exists(index=self.index_name):
            return False

        self.client.indices.create(
            index=self.index_name,
            body=build_knn_index_mapping(dimensions)
        )
        return True

    def add(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> None:
        self._validate_dimensions(embedding)
        self.client.index(
            index=self.index_name,
            id=chunk.chunk_id,
            body=self._document_body(chunk, embedding)
        )

    def add_many(
        self,
        records: list[tuple[Chunk, list[float]]]
    ) -> None:
        """
        Real bulk indexing via the OpenSearch _bulk API - one HTTP request
        for the whole batch instead of one request per chunk.
        """
        if not records:
            return

        body: list[dict[str, Any]] = []

        for chunk, embedding in records:
            self._validate_dimensions(embedding)
            body.append({"index": {"_index": self.index_name, "_id": chunk.chunk_id}})
            body.append(self._document_body(chunk, embedding))

        response = self.client.bulk(body=body)

        if response.get("errors"):
            failed = [
                item["index"]
                for item in response.get("items", [])
                if "index" in item and item["index"].get("error")
            ]
            raise BulkIndexError(failed)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        self._validate_dimensions(query_embedding)
        body = self._search_body(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        return self._results_from_response(response)

    def search_lexical(
        self,
        query_text: str,
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """
        Real BM25 search via OpenSearch's own match query against the text
        field - term frequency / inverse document frequency / length
        normalization all handled by the engine itself, unlike the
        regex-overlap approximation HybridRetriever uses today.
        """
        filters = self._metadata_filters(metadata_filter)
        query: dict[str, Any] = {
            "bool": {
                "must": [{"match": {"text": query_text}}],
                "filter": filters
            }
        }

        response = self.client.search(
            index=self.index_name,
            body={"size": top_k, "query": query}
        )
        return self._results_from_response(response)

    def get_embedding(
        self,
        chunk_id: str
    ) -> list[float] | None:
        """
        Fetches a previously stored chunk's raw embedding vector back
        out - used by IncrementalIndexer to reuse an unchanged chunk's
        embedding under a new chunk_id (its position on the page
        shifted but its content, and therefore its embedding, didn't)
        without a real embedding-model/API call.
        """
        response = self.client.get(index=self.index_name, id=chunk_id, ignore=[404])

        if not response.get("found", False):
            return None

        return response.get("_source", {}).get("embedding")

    def delete(
        self,
        chunk_id: str
    ) -> None:
        self.client.delete(
            index=self.index_name,
            id=chunk_id,
            ignore=[404]
        )

    def delete_by_document(
        self,
        document_id: str
    ) -> int:
        """
        Removes every chunk belonging to a document - used by document
        delete/reindex so no orphaned chunks are left behind. Returns how
        many chunks were actually deleted.

        conflicts="proceed" - verified against a real domain that
        _delete_by_query raises a 409 version_conflict_engine_exception
        when a chunk was updated (e.g. via update_metadata) in the same
        near-real-time indexing window as the delete - OpenSearch's
        default behavior is to abort the whole request on any conflict.
        Deleting a document's chunks should succeed regardless of a
        benign concurrent version bump; "proceed" skips conflicting docs
        rather than failing the entire delete.
        """
        response = self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"document_id": document_id}}},
            conflicts="proceed"
        )
        return response.get("deleted", 0)

    def update_metadata(
        self,
        chunk_id: str,
        metadata: dict[str, Any]
    ) -> None:
        """
        Partial update - only touches the metadata field, without
        re-sending the text or re-computing the embedding.
        """
        self.client.update(
            index=self.index_name,
            id=chunk_id,
            body={"doc": {"metadata": metadata}}
        )

    def count(self) -> int:
        """
        Real chunk count for this index - added after mypy caught a
        genuine production bug: the scheduled health-check job called
        len(vector_store), which only InMemoryVectorStore supports.
        OpenSearchVectorStore had no equivalent, so that job silently
        failed (caught by Scheduler._execute's own per-job try/except)
        every single run against any OpenSearch-backed deployment.
        """
        response = self.client.count(index=self.index_name)
        return response.get("count", 0)

    def health_check(self) -> dict[str, Any]:
        """
        Cluster-wide health, deliberately not scoped to self.index_name -
        verified against a real domain that GET /_cluster/health/{index}
        hangs indefinitely rather than erroring when that index doesn't
        exist yet, which would make a naive readiness check hang forever
        on a fresh deployment before the index is created.
        """
        return self.client.cluster.health()

    def _validate_dimensions(
        self,
        embedding: list[float]
    ) -> None:
        if self.embedding_dimensions is None:
            return

        if len(embedding) != self.embedding_dimensions:
            raise EmbeddingDimensionError(
                f"embedding has {len(embedding)} dimensions, expected "
                f"{self.embedding_dimensions} for index {self.index_name!r}"
            )

    def _document_body(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "document_version": chunk.document_version,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source": chunk.source,
            "document_type": chunk.document_type,
            "owner": chunk.owner,
            "parent_section": chunk.parent_section,
            "content_hash": chunk.content_hash,
            "chunking_version": chunk.chunking_version,
            "embedding_provider": chunk.embedding_provider,
            "embedding_model": chunk.embedding_model,
            "embedding_version": chunk.embedding_version,
            "tenant_id": chunk.tenant_id,
            "access_groups": chunk.access_groups,
            "classification": chunk.classification,
            "created_at": chunk.created_at.isoformat() if chunk.created_at else None,
            "indexed_at": chunk.indexed_at.isoformat() if chunk.indexed_at else None,
            "metadata": chunk.metadata,
            "embedding": embedding,
        }

    def _metadata_filters(
        self,
        metadata_filter: dict[str, str] | None
    ) -> list[dict[str, Any]]:
        return [
            {"term": {f"metadata.{key}": value}}
            for key, value in (metadata_filter or {}).items()
        ]

    def _search_body(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, str] | None
    ) -> dict[str, Any]:
        """
        OpenSearch's k-NN query DSL, not Elasticsearch's - the two diverged
        and use different field names for the same idea (OpenSearch:
        vector/k/filter nested under the mapped field's own key;
        Elasticsearch: field/query_vector/num_candidates as siblings).
        Verified against a real domain - the Elasticsearch-shaped body this
        used to send fails with "unknown field [field]".
        """
        filters = self._metadata_filters(metadata_filter)
        knn_clause: dict[str, Any] = {
            "vector": query_embedding,
            "k": top_k,
        }

        if filters:
            knn_clause["filter"] = {"bool": {"filter": filters}}

        return {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": knn_clause
                }
            }
        }

    def _results_from_response(
        self,
        response: dict[str, Any]
    ) -> list[SearchResult]:
        hits = response.get("hits", {}).get("hits", [])
        return [
            SearchResult(
                chunk=self._chunk_from_source(hit.get("_source", {})),
                score=float(hit.get("_score") or 0.0)
            )
            for hit in hits
        ]

    def _chunk_from_source(
        self,
        source: dict[str, Any]
    ) -> Chunk:
        return Chunk(
            chunk_id=source["chunk_id"],
            document_id=source["document_id"],
            document_version=source.get("document_version", 1),
            chunk_index=source["chunk_index"],
            text=source["text"],
            source=source["source"],
            document_type=source["document_type"],
            owner=source.get("owner"),
            parent_section=source.get("parent_section"),
            content_hash=source.get("content_hash"),
            chunking_version=source.get("chunking_version"),
            embedding_provider=source.get("embedding_provider"),
            embedding_model=source.get("embedding_model"),
            embedding_version=source.get("embedding_version"),
            tenant_id=source.get("tenant_id"),
            access_groups=source.get("access_groups") or [],
            classification=source.get("classification"),
            created_at=source.get("created_at"),
            indexed_at=source.get("indexed_at"),
            metadata=source.get("metadata", {}),
        )
