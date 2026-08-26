import json

from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from ingestion.sqs_ingestion_worker import SQSIngestionWorker
from mlops.ingestion_job_store import IngestionJobStore
from mlops.ingestion_job_store import JobStatus


class _NoSuchKey(Exception):
    pass


class _FakeExceptions:
    NoSuchKey = _NoSuchKey


class _FakeBody:

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


class FakeS3JobClient:
    """Fake S3 client backing IngestionJobStore for these worker tests."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.exceptions = _FakeExceptions()

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()

        return {"Body": _FakeBody(self.objects[Key])}


class FakeSQSClient:

    def __init__(self, messages=None):
        self._messages = messages or []
        self.deleted_receipt_handles = []

    def receive_message(self, QueueUrl, MaxNumberOfMessages, WaitTimeSeconds):
        messages, self._messages = self._messages, []
        return {"Messages": messages}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted_receipt_handles.append(ReceiptHandle)


class _FakeIngestionPipeline:

    def __init__(self, result=None):
        self.result = result or Result(
            success=True,
            data=Document(
                document_id="doc-1", source="s3://bucket/raw/doc-1.md",
                document_type="markdown", content="some content"
            )
        )
        self.calls = []

    def ingest_from_s3(self, s3_store, key, document_id):
        self.calls.append({"key": key, "document_id": document_id})
        return self.result


class _FakeS3Store:

    def __init__(self):
        self.bucket_name = "my-bucket"
        self.processed = []
        self.failed = []

    def mark_processed(self, key):
        self.processed.append(key)

    def mark_failed(self, key, reason=None):
        self.failed.append({"key": key, "reason": reason})


class _FakeRAGService:

    def __init__(self, chunk_count=3):
        self.chunk_count = chunk_count
        self.indexed_documents = []

    def index_document(self, document):
        self.indexed_documents.append(document)
        return self.chunk_count


def _sqs_message(job_id, document_id, key, receipt_handle="rh-1"):
    return {
        "ReceiptHandle": receipt_handle,
        "Body": json.dumps({"job_id": job_id, "document_id": document_id, "key": key})
    }


def test_poll_once_processes_a_message_and_marks_it_indexed():

    job_store = IngestionJobStore(client=FakeS3JobClient(), bucket_name="jobs-bucket")
    job_store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    sqs = FakeSQSClient([_sqs_message("job-1", "doc-1", "raw/doc-1.md")])
    pipeline = _FakeIngestionPipeline()
    rag_service = _FakeRAGService(chunk_count=5)
    s3_store = _FakeS3Store()
    worker = SQSIngestionWorker(
        sqs_client=sqs,
        queue_url="https://sqs.example/queue",
        ingestion_pipeline=pipeline,
        s3_store=s3_store,
        job_store=job_store,
        rag_service=rag_service
    )

    count = worker.poll_once()

    assert count == 1
    assert job_store.get_job("job-1")["status"] == JobStatus.INDEXED.value
    assert s3_store.processed == ["raw/doc-1.md"]
    assert sqs.deleted_receipt_handles == ["rh-1"]
    assert len(rag_service.indexed_documents) == 1


def test_poll_once_marks_failed_and_does_not_delete_message_on_error():

    job_store = IngestionJobStore(client=FakeS3JobClient(), bucket_name="jobs-bucket")
    job_store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")

    sqs = FakeSQSClient([_sqs_message("job-1", "doc-1", "raw/doc-1.md")])
    pipeline = _FakeIngestionPipeline(
        result=Result(success=False, error=Error(code="PARSE_ERROR", message="bad file"))
    )
    s3_store = _FakeS3Store()
    worker = SQSIngestionWorker(
        sqs_client=sqs,
        queue_url="https://sqs.example/queue",
        ingestion_pipeline=pipeline,
        s3_store=s3_store,
        job_store=job_store,
        rag_service=_FakeRAGService()
    )

    worker.poll_once()

    record = job_store.get_job("job-1")
    assert record["status"] == JobStatus.FAILED.value
    assert "bad file" in record["error"]
    assert s3_store.failed
    assert sqs.deleted_receipt_handles == []


def test_poll_once_skips_and_deletes_an_already_indexed_duplicate():

    job_store = IngestionJobStore(client=FakeS3JobClient(), bucket_name="jobs-bucket")
    job_store.create_job("job-1", document_id="doc-1", s3_key="raw/doc-1.md")
    job_store.update_status("job-1", JobStatus.INDEXED)

    sqs = FakeSQSClient([_sqs_message("job-1", "doc-1", "raw/doc-1.md")])
    pipeline = _FakeIngestionPipeline()
    rag_service = _FakeRAGService()
    worker = SQSIngestionWorker(
        sqs_client=sqs,
        queue_url="https://sqs.example/queue",
        ingestion_pipeline=pipeline,
        s3_store=_FakeS3Store(),
        job_store=job_store,
        rag_service=rag_service
    )

    worker.poll_once()

    assert pipeline.calls == []
    assert rag_service.indexed_documents == []
    assert sqs.deleted_receipt_handles == ["rh-1"]


def test_poll_once_with_no_messages_returns_zero():

    job_store = IngestionJobStore(client=FakeS3JobClient(), bucket_name="jobs-bucket")
    sqs = FakeSQSClient([])
    worker = SQSIngestionWorker(
        sqs_client=sqs,
        queue_url="https://sqs.example/queue",
        ingestion_pipeline=_FakeIngestionPipeline(),
        s3_store=_FakeS3Store(),
        job_store=job_store,
        rag_service=_FakeRAGService()
    )

    assert worker.poll_once() == 0
