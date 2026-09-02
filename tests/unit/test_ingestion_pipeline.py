from ingestion.ingestion_pipeline import IngestionPipeline


def test_ingestion_pipeline_parses_and_cleans_markdown(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("# Policy\n\nEmployees    get leave.", encoding="utf-8")

    pipeline = IngestionPipeline()

    result = pipeline.ingest_file(str(file_path), document_id="policy")

    assert result.success is True
    assert result.data is not None
    assert result.data.content == "# Policy\nEmployees get leave."


def test_ingestion_pipeline_uses_the_given_document_id(tmp_path):

    file_path = tmp_path / "leave-policy.md"
    file_path.write_text("Employees get leave.", encoding="utf-8")

    result = IngestionPipeline().ingest_file(str(file_path), document_id="stable-id-123")

    assert result.data.document_id == "stable-id-123"


def test_ingestion_pipeline_explicit_document_id_survives_a_file_rename(tmp_path):
    """
    The whole point of document_id being mandatory rather than
    filename-derived: the same logical document, ingested under two
    different filenames, keeps the same document_id as long as the
    caller supplies the same stable id both times.
    """
    original = tmp_path / "leave-policy-v1.md"
    original.write_text("Employees get leave.", encoding="utf-8")
    renamed = tmp_path / "leave-policy-v2-renamed.md"
    renamed.write_text("Employees get leave.", encoding="utf-8")

    pipeline = IngestionPipeline()
    first = pipeline.ingest_file(str(original), document_id="stable-id-123")
    second = pipeline.ingest_file(str(renamed), document_id="stable-id-123")

    assert first.data.document_id == second.data.document_id == "stable-id-123"


def test_ingestion_pipeline_rejects_unsupported_file(tmp_path):

    file_path = tmp_path / "policy.txt"
    file_path.write_text("hello", encoding="utf-8")

    pipeline = IngestionPipeline()

    result = pipeline.ingest_file(str(file_path), document_id="policy")

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

    result = pipeline.ingest_from_s3(s3_store, key="raw/leave-policy.md", document_id="leave-policy")

    assert result.success is True
    assert not (download_dir / "downloaded.md").exists()


def test_ingest_from_s3_overrides_source_with_the_real_key(tmp_path):

    download_dir = tmp_path / "download"
    download_dir.mkdir()
    s3_store = _FakeS3DocumentStore(download_dir)
    pipeline = IngestionPipeline()

    result = pipeline.ingest_from_s3(s3_store, key="raw/leave-policy.md", document_id="leave-policy")

    assert result.data.document_id == "leave-policy"
    assert result.data.source == "s3://my-bucket/raw/leave-policy.md"
    assert result.data.metadata["file_name"] == "leave-policy.md"


def test_ingest_from_s3_uses_the_given_document_id(tmp_path):

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

    result = pipeline.ingest_from_s3(s3_store, key="raw/empty.md", document_id="empty")

    assert result.success is False
    assert not (download_dir / "downloaded.md").exists()
