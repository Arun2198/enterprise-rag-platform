from ingestion.contracts.document import Document
from rag.chunking.recursive_chunker import RecursiveChunker


def test_recursive_chunker_preserves_metadata_and_order():

    document = Document(
        document_id="doc-1",
        source="policy.md",
        document_type="markdown",
        content="# Leave Policy\nEmployees get 20 days. Contractors get 10 days.",
        owner="HR",
        metadata={"category": "policy"}
    )
    chunker = RecursiveChunker(chunk_size=45, chunk_overlap=10, minimum_chunk_size=5)

    result = chunker.chunk(document)

    assert result.success is True
    assert result.data is not None
    assert len(result.data) >= 1
    assert result.data[0].chunk_index == 0
    assert result.data[0].metadata["category"] == "policy"
    assert result.data[0].metadata["document_id"] == "doc-1"
