from typing import Any

from rag.chunking.chunk import Chunk
from rag.vector_store.in_memory_store import SearchResult


class OpenSearchVectorStore:
    """
    OpenSearch adapter boundary.

    The client is injected to keep the core package testable and to avoid
    importing AWS/OpenSearch libraries until deployment packaging is configured.
    """

    def __init__(
        self,
        client: Any,
        index_name: str
    ) -> None:
        self.client = client
        self.index_name = index_name

    def add(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> None:
        self.client.index(
            index=self.index_name,
            id=chunk.chunk_id,
            body=self._document_body(chunk, embedding)
        )

    def add_many(
        self,
        records: list[tuple[Chunk, list[float]]]
    ) -> None:
        for chunk, embedding in records:
            self.add(chunk, embedding)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, str] | None = None
    ) -> list[SearchResult]:
        body = self._search_body(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        response = self.client.search(
            index=self.index_name,
            body=body
        )
        hits = response.get("hits", {}).get("hits", [])
        results = []

        for hit in hits:
            source = hit.get("_source", {})
            results.append(
                SearchResult(
                    chunk=self._chunk_from_source(source),
                    score=float(hit.get("_score") or 0.0)
                )
            )

        return results

    def _document_body(
        self,
        chunk: Chunk,
        embedding: list[float]
    ) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source": chunk.source,
            "document_type": chunk.document_type,
            "owner": chunk.owner,
            "parent_section": chunk.parent_section,
            "metadata": chunk.metadata,
            "embedding": embedding,
        }

    def _search_body(
        self,
        query_embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, str] | None
    ) -> dict[str, Any]:
        filters = []

        for key, value in (metadata_filter or {}).items():
            filters.append({"term": {f"metadata.{key}": value}})

        knn_query: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": top_k,
            "num_candidates": max(top_k * 10, 100),
        }

        if filters:
            knn_query["filter"] = {"bool": {"filter": filters}}

        return {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": knn_query
                }
            }
        }

    def _chunk_from_source(
        self,
        source: dict[str, Any]
    ) -> Chunk:
        return Chunk(
            chunk_id=source["chunk_id"],
            document_id=source["document_id"],
            chunk_index=source["chunk_index"],
            text=source["text"],
            source=source["source"],
            document_type=source["document_type"],
            owner=source.get("owner"),
            parent_section=source.get("parent_section"),
            metadata=source.get("metadata", {}),
        )
