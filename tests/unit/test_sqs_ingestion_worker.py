import json
import os
import tempfile

from app.services.rag_service import RAGService
from ingestion.contracts.document import Document
from ingestion.contracts.result import Error
from ingestion.contracts.result import Result
from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.manifest_store import InMemoryManifestStore
from ingestion.sqs_ingestion_worker import SQSIngestionWorker
from mlops.ingestion_job_store import IngestionJobStore
from mlops.ingestion_job_store import JobStatus
from rag.chunking.recursive_chunker import RecursiveChunker
from rag.embeddings.hashing_embedder import HashingEmbedder
from rag.vector_store.in_memory_store import InMemoryVectorStore


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


class _RealContentS3Store:
    """
    Backs a real IngestionPipeline with actual file content (unlike
    _FakeS3Store above, which never gets past a canned Document) - what
    the two integration tests below need to exercise real parsing +
    chunking + IncrementalIndexer, not a mocked-out pipeline.
    """

    def __init__(self, content_by_key: dict[str, str]):
        self.bucket_name = "my-bucket"
        self._content_by_key = content_by_key
        self.processed: list[str] = []
        self.failed: list[dict] = []

    def download_to_temp(self, key: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(self._content_by_key[key])
        return path

    def mark_processed(self, key):
        self.processed.append(key)

    def mark_failed(self, key, reason=None):
        self.failed.append({"key": key, "reason": reason})


def _build_real_worker(content_by_key, messages, rag_service=None, job_store=None):
    rag_service = rag_service or RAGService(
        embedder=HashingEmbedder(),
        vector_store=InMemoryVectorStore(),
        manifest_store=InMemoryManifestStore(),
        chunker=RecursiveChunker(chunk_size=900, chunk_overlap=50, minimum_chunk_size=10)
    )
    job_store = job_store or IngestionJobStore(client=FakeS3JobClient(), bucket_name="jobs-bucket")
    sqs = FakeSQSClient(messages)
    s3_store = _RealContentS3Store(content_by_key)
    worker = SQSIngestionWorker(
        sqs_client=sqs,
        queue_url="https://sqs.example/queue",
        ingestion_pipeline=IngestionPipeline(),
        s3_store=s3_store,
        job_store=job_store,
        rag_service=rag_service
    )
    return worker, rag_service, job_store, sqs


def test_uploading_the_same_document_id_twice_routes_through_incremental_diff_not_duplicate():
    content = "# Policy\nContractors receive 10 days of leave per year."
    worker, rag_service, job_store, sqs = _build_real_worker(
        content_by_key={"raw/doc-x.md": content},
        messages=[_sqs_message("job-1", "doc-x", "raw/doc-x.md")]
    )
    job_store.create_job("job-1", document_id="doc-x", s3_key="raw/doc-x.md")

    worker.poll_once()
    count_after_first = rag_service.vector_store.count()
    assert count_after_first > 0

    # Same document_id, same content, a new upload (new job_id) - the
    # real-world "re-upload to update" case, not a redelivered duplicate
    # message (that's the job-level dedup already covered above).
    sqs._messages = [_sqs_message("job-2", "doc-x", "raw/doc-x.md")]
    job_store.create_job("job-2", document_id="doc-x", s3_key="raw/doc-x.md")

    worker.poll_once()

    assert rag_service.vector_store.count() == count_after_first  # no duplicate chunks
    manifest = rag_service.manifest_store.get("doc-x")
    assert manifest.document_version == 1  # content unchanged - no version bump either


def test_uploading_two_different_document_ids_creates_two_separate_documents():
    worker, rag_service, job_store, sqs = _build_real_worker(
        content_by_key={
            "raw/doc-a.md": "# Doc A\nContent that belongs only to document A.",
            "raw/doc-b.md": "# Doc B\nCompletely different content for document B.",
        },
        messages=[_sqs_message("job-1", "doc-a", "raw/doc-a.md")]
    )
    job_store.create_job("job-1", document_id="doc-a", s3_key="raw/doc-a.md")
    worker.poll_once()

    sqs._messages = [_sqs_message("job-2", "doc-b", "raw/doc-b.md")]
    job_store.create_job("job-2", document_id="doc-b", s3_key="raw/doc-b.md")
    worker.poll_once()

    doc_ids_in_store = {
        chunk.document_id
        for chunk, _ in rag_service.vector_store._records.values()
    }
    assert doc_ids_in_store == {"doc-a", "doc-b"}
    assert rag_service.manifest_store.get("doc-a") is not None
    assert rag_service.manifest_store.get("doc-b") is not None
