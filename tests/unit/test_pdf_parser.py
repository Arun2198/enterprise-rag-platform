from ingestion.parsers.pdf_parser import PDFParser


def test_pdf_parser_success(tmp_path):

    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"5 0 obj << /Length 44 >> stream\n"
        b"BT /F1 12 Tf 72 720 Td (Hello PDF world) Tj ET\n"
        b"endstream endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000262 00000 n \n"
        b"0000000332 00000 n \n"
        b"trailer << /Root 1 0 R /Size 6 >>\n"
        b"startxref\n426\n%%EOF\n"
    )

    parser = PDFParser()

    result = parser.parse(str(file_path))

    assert result.success is True
    assert result.data is not None
    assert "Hello PDF world" in result.data.content


def test_pdf_parser_missing_file():

    parser = PDFParser()

    result = parser.parse(
        "sample_documents/does_not_exist.pdf"
    )

    assert result.success is False
    assert result.error is not None
