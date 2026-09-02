import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.config import load_settings
from app.service_factory import build_rag_service


def main():
    # goes through the real factory rather than a bare RAGService() so the
    # demo run uses the same quality embedding model (and guardrails,
    # reranker) as the deployed app - never the hashing baseline
    service = build_rag_service(load_settings())
    ingest_response = service.ingest(
        ["sample_documents/AI-RMF-1stdraft.pdf"],
        document_ids=["AI-RMF-1stdraft"]
    )

    print(f"Indexed documents: {ingest_response.indexed_documents}")
    print(f"Indexed chunks: {ingest_response.indexed_chunks}")

    if ingest_response.errors:
        print("Errors:")
        for error in ingest_response.errors:
            print(f"- {error}")

    response = service.ask("What is AI risk management?", top_k=3)
    print("\nAnswer:")
    print(response.answer)
    print("\nSources:")
    for source in response.sources:
        print(f"- {source.document_id} / {source.chunk_id} ({source.score:.3f})")


if __name__ == "__main__":
    main()
