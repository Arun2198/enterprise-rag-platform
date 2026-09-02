from ingestion.parsers.pdf_parser import PDFParser


def test_pdf_parser_success():

    parser = PDFParser()

    result = parser.parse(
        "sample_documents/AI-RMF-1stdraft.pdf"
    )

    assert result.success is True
    assert result.data is not None
    assert len(result.data.content) > 0


def test_pdf_parser_missing_file():

    parser = PDFParser()

    result = parser.parse(
        "sample_documents/does_not_exist.pdf"
    )

    assert result.success is False
    assert result.error is not None


def test_pdf_parser_populates_per_page_text():

    parser = PDFParser()

    result = parser.parse("sample_documents/AI-RMF-1stdraft.pdf")

    assert result.success is True
    assert result.data.pages is not None
    assert len(result.data.pages) == result.data.metadata["page_count"]
    # page list stays aligned to the real page count even if any single
    # page had no extractable text - never silently compacted
    assert all(isinstance(page, str) for page in result.data.pages)
    assert "".join(result.data.pages).strip() != ""
