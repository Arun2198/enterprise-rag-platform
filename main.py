import sys
from tempfile import TemporaryDirectory
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.services.rag_service import RAGService


def main():
    service = RAGService()
    sample_pdf = Path("sample_documents/AI-RMF-1stdraft.pdf")

    if sample_pdf.exists():
        file_paths = [str(sample_pdf)]
        query = "What is AI risk management?"
        metadata = {"domain": "ai_governance"}
        ingest_response = service.ingest(file_paths, metadata=metadata)
    else:
        with TemporaryDirectory() as temp_dir:
            demo_file = Path(temp_dir) / "leave_policy.md"
            demo_file.write_text(
                "Employees receive 20 days of paid leave annually. "
                "Contractors receive 10 days of leave.",
                encoding="utf-8"
            )
            query = "How many leave days do contractors receive?"
            ingest_response = service.ingest(
                [str(demo_file)],
                metadata={"domain": "demo", "department": "hr"}
            )
            _print_run(service, ingest_response, query)
            return

    _print_run(service, ingest_response, query)


def _print_run(
    service: RAGService,
    ingest_response,
    query: str
) -> None:

    print(f"Indexed documents: {ingest_response.indexed_documents}")
    print(f"Indexed chunks: {ingest_response.indexed_chunks}")

    if ingest_response.errors:
        print("Errors:")
        for error in ingest_response.errors:
            print(f"- {error}")

    response = service.ask(query, top_k=3)
    print("\nAnswer:")
    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        print(f"- {source.document_id} / {source.chunk_id} ({source.score:.3f})")


if __name__ == "__main__":
    main()
