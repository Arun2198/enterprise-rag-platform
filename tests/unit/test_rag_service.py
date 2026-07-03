from app.services.rag_service import RAGService
from rag.chunking.recursive_chunker import RecursiveChunker


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


def test_rag_service_filters_by_metadata(tmp_path):

    hr_file = tmp_path / "hr_policy.md"
    finance_file = tmp_path / "finance_policy.md"
    hr_file.write_text("Contractors receive 10 days of leave.", encoding="utf-8")
    finance_file.write_text("Invoices are paid within 30 days.", encoding="utf-8")
    service = RAGService()

    service.ingest([str(hr_file)], metadata={"department": "hr"})
    service.ingest([str(finance_file)], metadata={"department": "finance"})
    response = service.ask(
        query="How many leave days do contractors receive?",
        metadata_filter={"department": "hr"}
    )

    assert response.sources
    assert all(
        "hr_policy" in source.document_id
        for source in response.sources
    )
