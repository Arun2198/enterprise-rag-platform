import json
import logging
from typing import Any

from mlops.scheduler import JobNotFoundError
from mlops.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SQSSchedulerWorker:
    """
    EventBridge Scheduler -> SQS -> worker job execution.

    Fixes a real bug in plain interval-based scheduling: with N ECS
    tasks each running its own asyncio loop calling
    Scheduler.run_due_jobs() on the same interval, every task
    independently fires every due job every interval - a "backup" or
    "health_check" job registered once still runs N times per interval,
    once per task, since each task holds its own in-memory Scheduler
    with no coordination between them. Routing the trigger through SQS
    fixes this for free: SQS delivers each message to exactly one
    consumer at a time no matter how many tasks are polling the same
    queue, so exactly one task executes each scheduled run - the same
    single-delivery property SQSIngestionWorker already relies on.

    One aws_scheduler_schedule (EventBridge Scheduler) per job sends a
    {"job_id": ...} message to the queue on its own cron/rate
    expression. This worker only executes the matching *registered*
    job by id - it does not do its own interval bookkeeping.
    Scheduler.trigger() still updates next_run_at as a side effect
    (kept for /admin/scheduler/jobs visibility and job-history
    consistency), but EventBridge - not that timestamp - is the actual
    source of truth for when a job next runs in this mode.
    """

    def __init__(
        self,
        sqs_client: Any,
        queue_url: str,
        scheduler: Scheduler,
        max_messages: int = 10,
        wait_time_seconds: int = 10
    ) -> None:
        self.sqs_client = sqs_client
        self.queue_url = queue_url
        self.scheduler = scheduler
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

        try:
            self.scheduler.trigger(job_id)
        except JobNotFoundError:
            logger.warning("scheduled_job_not_registered", extra={"job_id": job_id})
        except Exception as ex:
            # Scheduler._execute already catches exceptions from the job's
            # own callable and records them as a failed JobRun - this
            # branch is only for something unexpected escaping trigger()
            # itself.
            logger.warning(
                "scheduled_job_execution_error",
                extra={"job_id": job_id, "error": str(ex)}
            )
        finally:
            # Always delete, including on failure - a failed backup/health
            # check is already recorded via JobRun.success=False, and
            # EventBridge will fire the next scheduled message on its own
            # cron regardless. Leaving this one for SQS redelivery would
            # just re-run a job that already ran (successfully or not)
            # this cycle, which is exactly the duplicate-execution problem
            # this worker exists to avoid.
            self._delete_message(receipt_handle)

    def _delete_message(
        self,
        receipt_handle: str
    ) -> None:
        self.sqs_client.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle
        )
