import os

import pytest

from ingestion.s3_document_store import S3DocumentStore
from ingestion.s3_document_store import S3ValidationError


class FakeS3Client:

    def __init__(self):
        self.uploaded = []
        self.downloaded = []
        self.copied = []
        self.deleted = []
        self.head_object_metadata = {}

    def upload_file(self, local_path, bucket, key, ExtraArgs=None):
        self.uploaded.append({"local_path": local_path, "bucket": bucket, "key": key, "extra": ExtraArgs})

    def download_file(self, bucket, key, local_path):
        self.downloaded.append({"bucket": bucket, "key": key, "local_path": local_path})
        with open(local_path, "w", encoding="utf-8") as f:
            f.write("downloaded content")

    def copy_object(self, **kwargs):
        self.copied.append(kwargs)

    def delete_object(self, Bucket, Key):
        self.deleted.append({"bucket": Bucket, "key": Key})

    def head_object(self, Bucket, Key):
        return {"Metadata": self.head_object_metadata.get(Key, {})}


def test_upload_validates_and_uploads_to_the_raw_prefix(tmp_path):

    local_file = tmp_path / "policy.md"
    local_file.write_text("some content", encoding="utf-8")
    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    key = store.upload(str(local_file), document_id="doc-1")

    assert key == "raw/doc-1.md"
    assert client.uploaded[0]["bucket"] == "my-bucket"
    assert client.uploaded[0]["extra"]["Metadata"]["original_filename"] == "policy.md"
    assert client.uploaded[0]["extra"]["Metadata"]["document_id"] == "doc-1"


def test_upload_rejects_a_disallowed_file_type(tmp_path):

    local_file = tmp_path / "malware.exe"
    local_file.write_text("x", encoding="utf-8")
    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    with pytest.raises(S3ValidationError):
        store.upload(str(local_file), document_id="doc-1")


def test_upload_rejects_a_file_over_the_size_limit(tmp_path):

    local_file = tmp_path / "big.md"
    local_file.write_text("x" * 100, encoding="utf-8")
    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket", max_file_size_bytes=50)

    with pytest.raises(S3ValidationError):
        store.upload(str(local_file), document_id="doc-1")


def test_upload_rejects_an_empty_file(tmp_path):

    local_file = tmp_path / "empty.md"
    local_file.write_text("", encoding="utf-8")
    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    with pytest.raises(S3ValidationError):
        store.upload(str(local_file), document_id="doc-1")


def test_download_to_temp_writes_a_local_file_with_the_right_extension():

    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    local_path = store.download_to_temp("raw/doc-1.md")

    try:
        assert local_path.endswith(".md")
        assert os.path.exists(local_path)
        assert client.downloaded[0]["key"] == "raw/doc-1.md"
    finally:
        os.remove(local_path)


def test_mark_processed_copies_to_the_processed_prefix_and_deletes_the_original():

    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    new_key = store.mark_processed("raw/doc-1.md")

    assert new_key == "processed/doc-1.md"
    assert client.copied[0]["Key"] == "processed/doc-1.md"
    assert client.deleted[0]["key"] == "raw/doc-1.md"


def test_mark_failed_copies_to_the_failed_prefix_with_a_reason():

    client = FakeS3Client()
    store = S3DocumentStore(client=client, bucket_name="my-bucket")

    new_key = store.mark_failed("raw/doc-1.md", reason="unsupported encoding")

    assert new_key == "failed/doc-1.md"
    assert client.copied[0]["Metadata"]["failure_reason"] == "unsupported encoding"
    assert client.copied[0]["MetadataDirective"] == "REPLACE"
