import json
import logging
from datetime import datetime
from datetime import timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class IngestionJobStore:
    """
    Durable job status tracking for async ingestion, backed by S3 rather
    than introducing a new database - one small JSON object per job
    (jobs/{job_id}.json). This is genuinely required durable state (a
    caller needs to poll job status across worker restarts, and the API
    that enqueues a job returns before it's processed), but doesn't
    justify DynamoDB/RDS at this scale: S3 read-after-write consistency
    is sufficient for "check status of the job I just created," and it's
    one less AWS service to provision and pay for.
    """

    def __init__(
        self,
        client: Any,
        bucket_name: str,
        prefix: str = "jobs/"
    ) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.prefix = prefix

    def create_job(
        self,
        job_id: str,
        document_id: str,
        s3_key: str
    ) -> dict[str, Any]:
        record = {
            "job_id": job_id,
            "document_id": document_id,
            "s3_key": s3_key,
            "status": JobStatus.RECEIVED.value,
            "error": None,
            "created_at": self._now(),
            "updated_at": self._now()
        }
        self._write(job_id, record)
        return record

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: str | None = None
    ) -> dict[str, Any]:
        record = self.get_job(job_id)

        if record is None:
            raise KeyError(f"no job found for job_id={job_id!r}")

        record["status"] = status.value
        record["error"] = error
        record["updated_at"] = self._now()
        self._write(job_id, record)
        return record

    def get_job(
        self,
        job_id: str
    ) -> dict[str, Any] | None:
        try:
            response = self.client.get_object(Bucket=self.bucket_name, Key=self._key(job_id))
        except self.client.exceptions.NoSuchKey:
            return None

        return json.loads(response["Body"].read())

    def is_already_processed(
        self,
        job_id: str
    ) -> bool:
        """
        Duplicate-event protection: if this exact job_id was already
        INDEXED (or is currently PROCESSING), a redelivered SQS message
        for it should be a no-op rather than re-running ingestion and
        creating duplicate work. Chunk ids are also deterministic
        (document_id:index), so even a genuine re-run would overwrite
        rather than duplicate - this check just avoids the wasted work.
        """
        record = self.get_job(job_id)
        return record is not None and record["status"] in (
            JobStatus.PROCESSING.value, JobStatus.INDEXED.value
        )

    def _write(
        self,
        job_id: str,
        record: dict[str, Any]
    ) -> None:
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=self._key(job_id),
            Body=json.dumps(record).encode("utf-8"),
            ContentType="application/json"
        )

    def _key(
        self,
        job_id: str
    ) -> str:
        return f"{self.prefix}{job_id}.json"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
