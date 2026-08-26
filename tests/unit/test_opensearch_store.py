import pytest

from rag.chunking.chunk import Chunk
from rag.vector_store.opensearch_store import BulkIndexError
from rag.vector_store.opensearch_store import EmbeddingDimensionError
from rag.vector_store.opensearch_store import OpenSearchVectorStore


class FakeIndicesClient:

    def __init__(self):
        self.existing_indexes: set[str] = set()
        self.created_indexes: list[dict] = []

    def exists(self, index):
        return index in self.existing_indexes

    def create(self, index, body):
        self.existing_indexes.add(index)
        self.created_indexes.append({"index": index, "body": body})


class FakeClusterClient:

    def __init__(self, health_response=None):
        self._health_response = health_response or {"status": "green", "timed_out": False}
        self.health_calls = []

    def health(self, index=None):
        self.health_calls.append(index)
        return self._health_response


class FakeOpenSearchClient:

    def __init__(self, health_response=None):
        self.index_calls = []
        self.search_calls = []
        self.bulk_calls = []
        self.delete_calls = []
        self.delete_by_query_calls = []
        self.update_calls = []
        self.bulk_errors = False
        self.indices = FakeIndicesClient()
        self.cluster = FakeClusterClient(health_response)

    def index(self, index, id, body):
        self.index_calls.append({"index": index, "id": id, "body": body})

    def bulk(self, body):
        self.bulk_calls.append(body)

        if self.bulk_errors:
            return {
                "errors": True,
                "items": [
                    {"index": {"_id": "bad:0", "error": {"type": "mapper_parsing_exception"}}}
                ]
            }

        return {"errors": False, "items": []}

    def search(self, index, body):
        self.search_calls.append({"index": index, "body": body})
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 0.9,
                        "_source": {
                            "chunk_id": "doc:0",
                            "document_id": "doc",
                            "chunk_index": 0,
                            "text": "hello",
                            "source": "doc.md",
                            "document_type": "markdown",
                            "metadata": {"domain": "ai_governance"},
                        },
                    }
                ]
            }
        }

    def delete(self, index, id, ignore=None):
        self.delete_calls.append({"index": index, "id": id, "ignore": ignore})

    def delete_by_query(self, index, body, conflicts=None):
        self.delete_by_query_calls.append({"index": index, "body": body, "conflicts": conflicts})
        return {"deleted": 2}

    def update(self, index, id, body):
        self.update_calls.append({"index": index, "id": id, "body": body})


def _chunk():
    return Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="hello",
        source="doc.md",
        document_type="markdown",
    )


def test_opensearch_store_indexes_and_searches():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    store.add(_chunk(), [0.1, 0.2])
    results = store.search([0.1, 0.2], metadata_filter={"domain": "ai_governance"})

    assert client.index_calls[0]["id"] == "doc:0"
    assert client.search_calls[0]["body"]["query"]["knn"]["embedding"]["k"] == 5
    assert results[0].chunk.chunk_id == "doc:0"
    assert results[0].score == 0.9


def test_add_many_uses_the_real_bulk_api_in_one_request():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")
    records = [(_chunk(), [0.1, 0.2]), (_chunk(), [0.3, 0.4])]

    store.add_many(records)

    assert len(client.bulk_calls) == 1
    body = client.bulk_calls[0]
    assert len(body) == 4
    assert body[0] == {"index": {"_index": "chunks", "_id": "doc:0"}}


def test_add_many_is_a_no_op_for_an_empty_batch():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    store.add_many([])

    assert client.bulk_calls == []


def test_add_many_raises_bulk_index_error_on_partial_failure():

    client = FakeOpenSearchClient()
    client.bulk_errors = True
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    with pytest.raises(BulkIndexError):
        store.add_many([(_chunk(), [0.1, 0.2])])


def test_search_lexical_runs_a_real_match_query():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    results = store.search_lexical("hello world", top_k=3, metadata_filter={"domain": "ai_governance"})

    query = client.search_calls[0]["body"]["query"]
    assert query["bool"]["must"] == [{"match": {"text": "hello world"}}]
    assert query["bool"]["filter"] == [{"term": {"metadata.domain": "ai_governance"}}]
    assert results[0].chunk.chunk_id == "doc:0"


def test_delete_removes_a_single_chunk():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    store.delete("doc:0")

    assert client.delete_calls[0] == {"index": "chunks", "id": "doc:0", "ignore": [404]}


def test_delete_by_document_removes_every_chunk_for_that_document():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    deleted_count = store.delete_by_document("doc")

    call = client.delete_by_query_calls[0]
    assert call["body"]["query"]["term"]["document_id"] == "doc"
    assert call["conflicts"] == "proceed"
    assert deleted_count == 2


def test_update_metadata_sends_a_partial_update():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    store.update_metadata("doc:0", {"classification": "internal"})

    call = client.update_calls[0]
    assert call["id"] == "doc:0"
    assert call["body"] == {"doc": {"metadata": {"classification": "internal"}}}


def test_health_check_returns_the_cluster_health_response():

    client = FakeOpenSearchClient(health_response={"status": "yellow", "timed_out": False})
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    health = store.health_check()

    assert health["status"] == "yellow"
    assert client.cluster.health_calls == [None]


def test_ensure_index_creates_the_index_with_a_knn_mapping_when_missing():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    created = store.ensure_index(embedding_dimensions=384)

    assert created is True
    mapping = client.indices.created_indexes[0]["body"]
    assert mapping["settings"]["index"]["knn"] is True
    assert mapping["mappings"]["properties"]["embedding"]["dimension"] == 384


def test_ensure_index_is_a_no_op_when_the_index_already_exists():

    client = FakeOpenSearchClient()
    client.indices.existing_indexes.add("chunks")
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    created = store.ensure_index(embedding_dimensions=384)

    assert created is False
    assert client.indices.created_indexes == []


def test_ensure_index_requires_a_known_dimension():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    with pytest.raises(EmbeddingDimensionError):
        store.ensure_index()


def test_add_rejects_an_embedding_with_the_wrong_dimension():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks", embedding_dimensions=384)

    with pytest.raises(EmbeddingDimensionError):
        store.add(_chunk(), [0.1, 0.2])


def test_search_rejects_a_query_embedding_with_the_wrong_dimension():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks", embedding_dimensions=384)

    with pytest.raises(EmbeddingDimensionError):
        store.search([0.1, 0.2])


def test_add_allows_any_dimension_when_none_is_configured():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")

    store.add(_chunk(), [0.1, 0.2])

    assert client.index_calls[0]["id"] == "doc:0"
