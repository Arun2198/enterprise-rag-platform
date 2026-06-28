from ingestion.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline_parses_and_cleans_markdown(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\n\nEmployees    get leave.", encoding="utf-8")

    pipeline = IngestionPipeline()

    result = pipeline.ingest_file(str(file_path))

    assert result.success is True
    assert result.data is not None
    assert result.data.content == "# Policy\nEmployees get leave."


def test_ingestion_pipeline_rejects_unsupported_file(tmp_path):

    file_path = tmp_path / "policy.txt"
    file_path.write_text("hello", encoding="utf-8")

    pipeline = IngestionPipeline()

    result = pipeline.ingest_file(str(file_path))

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "UNSUPPORTED_FILE_TYPE"
