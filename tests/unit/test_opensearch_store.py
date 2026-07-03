from rag.chunking.chunk import Chunk
from rag.vector_store.opensearch_store import OpenSearchVectorStore


class FakeOpenSearchClient:

    def __init__(self):
        self.indices = FakeIndices()
        self.index_calls = []
        self.search_calls = []

    def index(self, index, id, body, refresh=False):
        self.index_calls.append(
            {
                "index": index,
                "id": id,
                "body": body,
                "refresh": refresh
            }
        )

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


class FakeIndices:

    def __init__(self):
        self.exists_calls = []
        self.create_calls = []

    def exists(self, index):
        self.exists_calls.append(index)
        return False

    def create(self, index, body):
        self.create_calls.append({"index": index, "body": body})


def test_opensearch_store_indexes_and_searches():

    client = FakeOpenSearchClient()
    store = OpenSearchVectorStore(client=client, index_name="chunks")
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="hello",
        source="doc.md",
        document_type="markdown",
    )

    store.add(chunk, [0.1, 0.2])
    results = store.search([0.1, 0.2], metadata_filter={"domain": "ai_governance"})

    assert client.indices.create_calls[0]["index"] == "chunks"
    mapping = client.indices.create_calls[0]["body"]["mappings"]["properties"]
    assert mapping["embedding"]["type"] == "knn_vector"
    assert mapping["embedding"]["dimension"] == 384
    assert client.index_calls[0]["id"] == "doc:0"
    assert client.index_calls[0]["refresh"] is False
    vector_query = client.search_calls[0]["body"]["query"]["knn"]["embedding"]
    assert vector_query["k"] == 5
    assert vector_query["vector"] == [0.1, 0.2]
    assert results[0].chunk.chunk_id == "doc:0"
    assert results[0].score == 0.9
