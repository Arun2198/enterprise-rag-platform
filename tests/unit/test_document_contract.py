from ingestion.contracts.document import Document


def test_document_contract_creation():

    document = Document(
        document_id="1",
        source="sample.pdf",
        document_type="policy",
        content="hello world"
    )

    assert document.document_id == "1"
    assert document.source == "sample.pdf"