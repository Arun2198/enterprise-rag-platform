from app.services.rag_service import RAGService
from rag.chunking.chunk import Chunk
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.retrieval.hybrid_retrieval import RetrievedChunk


def test_rag_service_ingests_and_answers_from_markdown(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text(
        "# Leave Policy\n"
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave.",
        encoding="utf-8"
    )
    service = RAGService(
        chunker=RecursiveChunker(
            chunk_size=120,
            chunk_overlap=20,
            minimum_chunk_size=10
        )
    )

    ingest_response = service.ingest([str(file_path)])
    ask_response = service.ask("How many leave days do contractors receive?")

    assert ingest_response.indexed_documents == 1
    assert ingest_response.indexed_chunks >= 1
    assert ingest_response.errors == []
    assert "Contractors receive 10 days of leave." in ask_response.answer
    assert ask_response.sources


def test_ask_bypasses_reranking_when_no_reranker_configured():

    class StubRetriever:

        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, metadata_filter=None):
            self.calls.append(top_k)
            return []

    service = RAGService()
    service.retriever = StubRetriever()

    service.ask("query", top_k=5)

    assert service.retriever.calls == [5]


def test_ask_requests_top_k_times_candidate_multiplier_from_retriever():

    class StubRetriever:

        def __init__(self):
            self.calls = []

        def retrieve(self, query, top_k, metadata_filter=None):
            self.calls.append(top_k)
            return []

    class StubReranker:

        def rerank(self, query, candidates, top_k):
            return []

    service = RAGService(reranker=StubReranker(), candidate_multiplier=4)
    service.retriever = StubRetriever()

    service.ask("query", top_k=5)

    assert service.retriever.calls == [20]


def test_ask_forwards_reranked_chunks_unchanged_to_answerer(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\nSome policy content here.", encoding="utf-8")

    captured = {}

    class RecordingAnswerer:

        def answer(self, query, retrieved_chunks):
            captured["chunks"] = retrieved_chunks
            captured["query"] = query
            return "recorded answer"

    reranked_chunk = Chunk(
        chunk_id="fixed:0",
        document_id="fixed",
        chunk_index=0,
        text="fixed reranked text",
        source="fixed.md",
        document_type="markdown"
    )
    expected = [
        RetrievedChunk(
            chunk=reranked_chunk,
            vector_score=0.1,
            keyword_score=0.1,
            score=0.99
        )
    ]

    class StubReranker:

        def __init__(self):
            self.received_top_k = None

        def rerank(self, query, candidates, top_k):
            self.received_top_k = top_k
            return expected

    reranker = StubReranker()
    service = RAGService(
        answerer=RecordingAnswerer(),
        reranker=reranker,
        candidate_multiplier=4
    )
    service.ingest([str(file_path)])

    response = service.ask("policy question", top_k=3)

    assert captured["chunks"] == expected
    assert captured["query"] == "policy question"
    assert reranker.received_top_k == 3
    assert response.answer == "recorded answer"
