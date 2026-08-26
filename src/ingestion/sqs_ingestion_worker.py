import json
import logging
from typing import Any

from ingestion.ingestion_pipeline import IngestionPipeline
from ingestion.s3_document_store import S3DocumentStore
from mlops.ingestion_job_store import IngestionJobStore
from mlops.ingestion_job_store import JobStatus

logger = logging.getLogger(__name__)


class SQSIngestionWorker:
    """
    S3 -> SQS -> worker asynchronous ingestion. The API side uploads to S3,
    enqueues one SQS message per document, and returns immediately with
    {document_id, job_id, status: RECEIVED} rather than blocking on
    parse/chunk/embed/index for a potentially large document.

    Deliberately owns no polling thread/loop of its own - same
    caller-drives-the-clock philosophy as mlops.scheduler.Scheduler. Call
    poll_once() from whatever actually owns scheduling in a deployment (an
    asyncio loop, an ECS worker task, a Lambda triggered by SQS itself).

    Expected message body (JSON): {"job_id": ..., "document_id": ...,
    "key": ...} - "key" is the S3 object key under the raw/ prefix.
    """

    def __init__(
        self,
        sqs_client: Any,
        queue_url: str,
        ingestion_pipeline: IngestionPipeline,
        s3_store: S3DocumentStore,
        job_store: IngestionJobStore,
        rag_service: Any,
        max_messages: int = 10,
        wait_time_seconds: int = 10
    ) -> None:
        self.sqs_client = sqs_client
        self.queue_url = queue_url
        self.ingestion_pipeline = ingestion_pipeline
        self.s3_store = s3_store
        self.job_store = job_store
        self.rag_service = rag_service
        self.max_messages = max_messages
        self.wait_time_seconds = wait_time_seconds

    def poll_once(self) -> int:
        """Receives and processes up to max_messages. Returns how many
        messages were received (not necessarily how many succeeded)."""
        response = self.sqs_client.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=self.max_messages,
            WaitTimeSeconds=self.wait_time_seconds
        )
        messages = response.get("Messages", [])

        for message in messages:
            self._process_message(message)

        return len(messages)

    def _process_message(
        self,
        message: dict[str, Any]
    ) -> None:
        receipt_handle = message["ReceiptHandle"]
        body = json.loads(message["Body"])
        job_id = body["job_id"]

        if self.job_store.is_already_processed(job_id):
            logger.info(
                "ingestion_job_skipped_duplicate",
                extra={"job_id": job_id}
            )
            self._delete_message(receipt_handle)
            return

        self.job_store.update_status(job_id, JobStatus.PROCESSING)

        try:
            document_result = self.ingestion_pipeline.ingest_from_s3(
                self.s3_store,
                key=body["key"],
                document_id=body["document_id"]
            )

            if not document_result.success or document_result.data is None:
                error_message = (
                    document_result.error.message if document_result.error else "parsing failed"
                )
                raise RuntimeError(error_message)

            chunk_count = self.rag_service.index_document(document_result.data)

            if chunk_count is None:
                raise RuntimeError("chunking failed")

            self.s3_store.mark_processed(body["key"])
            self.job_store.update_status(job_id, JobStatus.INDEXED)
            self._delete_message(receipt_handle)

            logger.info(
                "ingestion_job_indexed",
                extra={"job_id": job_id, "document_id": body["document_id"], "chunk_count": chunk_count}
            )
        except Exception as ex:
            logger.warning(
                "ingestion_job_failed",
                extra={"job_id": job_id, "error": str(ex)}
            )
            self.job_store.update_status(job_id, JobStatus.FAILED, error=str(ex))
            self.s3_store.mark_failed(body["key"], reason=str(ex))
            # deliberately not deleting the message - let SQS's visibility
            # timeout expire so it's redelivered (up to the queue's own
            # maxReceiveCount), then routed to a DLQ by the queue's redrive
            # policy rather than reimplementing that here

    def _delete_message(
        self,
        receipt_handle: str
    ) -> None:
        self.sqs_client.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle
        )
