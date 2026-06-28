from ingestion.parsers.markdown_parser import MarkdownParser


def test_markdown_parser_success(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Leave Policy\n\nEmployees get 20 days.", encoding="utf-8")

    parser = MarkdownParser()

    result = parser.parse(str(file_path))

    assert result.success is True
    assert result.data is not None
    assert result.data.document_type == "markdown"
    assert "Employees get 20 days." in result.data.content


def test_markdown_parser_empty_file(tmp_path):

    file_path = tmp_path / "empty.md"
    file_path.write_text("", encoding="utf-8")

    parser = MarkdownParser()

    result = parser.parse(str(file_path))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "EMPTY_DOCUMENT"
