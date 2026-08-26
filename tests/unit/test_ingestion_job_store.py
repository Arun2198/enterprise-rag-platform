import json

import pytest

from mlops.ingestion_job_store import IngestionJobStore
from mlops.ingestion_job_store import JobStatus


class FakeBody:

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class NoSuchKey(Exception):
    pass


class FakeExceptions:
    NoSuchKey = NoSuchKey


class FakeS3Client:

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.exceptions = FakeExceptions()

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        return {"Body": FakeBody(self.objects[Key])}


def test_create_job_writes_a_received_status():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")

    record = store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    assert record["status"] == JobStatus.RECEIVED.value
    assert json.loads(client.objects["jobs/job-1.json"])["job_id"] == "job-1"


def test_get_job_returns_none_when_the_job_does_not_exist():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")

    assert store.get_job("missing") is None


def test_update_status_transitions_and_persists():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")
    store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    store.update_status("job-1", JobStatus.PROCESSING)
    store.update_status("job-1", JobStatus.INDEXED)

    record = store.get_job("job-1")
    assert record["status"] == JobStatus.INDEXED.value


def test_update_status_records_the_error_on_failure():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")
    store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    store.update_status("job-1", JobStatus.FAILED, error="parsing failed")

    record = store.get_job("job-1")
    assert record["status"] == JobStatus.FAILED.value
    assert record["error"] == "parsing failed"


def test_update_status_on_an_unknown_job_raises():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")

    with pytest.raises(KeyError):
        store.update_status("missing", JobStatus.PROCESSING)


def test_is_already_processed_is_false_for_a_new_job():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")
    store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    assert store.is_already_processed("job-1") is False


def test_is_already_processed_is_true_once_indexed():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")
    store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")
    store.update_status("job-1", JobStatus.INDEXED)

    assert store.is_already_processed("job-1") is True


def test_is_already_processed_is_false_after_failure():

    client = FakeS3Client()
    store = IngestionJobStore(client=client, bucket_name="my-bucket")
    store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")
    store.update_status("job-1", JobStatus.FAILED, error="boom")

    assert store.is_already_processed("job-1") is False
