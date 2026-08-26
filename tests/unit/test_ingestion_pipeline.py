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


class _FakeS3DocumentStore:

    def __init__(self, tmp_path, content="# Policy\n\nEmployees get leave."):
        self.bucket_name = "my-bucket"
        self._tmp_path = tmp_path
        self._content = content
        self.downloaded_keys = []

    def download_to_temp(self, key):
        self.downloaded_keys.append(key)
        local_path = self._tmp_path / "downloaded.md"
        local_path.write_text(self._content, encoding="utf-8")
        return str(local_path)


def test_ingest_from_s3_downloads_and_parses_then_cleans_up(tmp_path):

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    s3_store = _FakeS3DocumentStore(download_dir)
    pipeline = IngestionPipeline()

    result = pipeline.ingest_from_s3(s3_store, key="raw/leave-policy.md")

    assert result.success is True
    assert not (download_dir / "downloaded.md").exists()


def test_ingest_from_s3_overrides_document_id_and_source_with_the_real_key(tmp_path):

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    s3_store = _FakeS3DocumentStore(download_dir)
    pipeline = IngestionPipeline()

    result = pipeline.ingest_from_s3(s3_store, key="raw/leave-policy.md")

    assert result.data.document_id == "leave-policy"
    assert result.data.source == "s3://my-bucket/raw/leave-policy.md"
    assert result.data.metadata["file_name"] == "leave-policy.md"


def test_ingest_from_s3_accepts_an_explicit_document_id(tmp_path):

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    s3_store = _FakeS3DocumentStore(download_dir)
    pipeline = IngestionPipeline()

    result = pipeline.ingest_from_s3(s3_store, key="raw/leave-policy.md", document_id="custom-id")

    assert result.data.document_id == "custom-id"


def test_ingest_from_s3_still_cleans_up_the_temp_file_on_parse_failure(tmp_path):

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    s3_store = _FakeS3DocumentStore(download_dir, content="")
    pipeline = IngestionPipeline()

    result = pipeline.ingest_from_s3(s3_store, key="raw/empty.md")

    assert result.success is False
    assert not (download_dir / "downloaded.md").exists()
