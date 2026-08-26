import json

from mlops.scheduler import Scheduler
from mlops.sqs_scheduler_worker import SQSSchedulerWorker


class FakeSQSClient:

    def __init__(self, messages=None):
        self._messages = messages or []
        self.deleted_receipt_handles = []

    def receive_message(self, QueueUrl, MaxNumberOfMessages, WaitTimeSeconds):
        messages, self._messages = self._messages, []
        return {"Messages": messages}

    def delete_message(self, QueueUrl, ReceiptHandle):
        self.deleted_receipt_handles.append(ReceiptHandle)


def _message(receipt_handle: str, job_id: str) -> dict:
    return {
        "ReceiptHandle": receipt_handle,
        "Body": json.dumps({"job_id": job_id})
    }


def test_poll_once_triggers_the_named_job():

    scheduler = Scheduler()
    calls = []
    scheduler.register(job_id="backup", name="Backup", interval_seconds=300, callable_=lambda: calls.append(1))
    client = FakeSQSClient(messages=[_message("r1", "backup")])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    processed = worker.poll_once()

    assert processed == 1
    assert calls == [1]
    assert scheduler.history("backup")[-1].success is True


def test_poll_once_deletes_message_after_successful_trigger():

    scheduler = Scheduler()
    scheduler.register(job_id="backup", name="Backup", interval_seconds=300, callable_=lambda: None)
    client = FakeSQSClient(messages=[_message("r1", "backup")])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    worker.poll_once()

    assert client.deleted_receipt_handles == ["r1"]


def test_poll_once_deletes_message_even_when_the_jobs_own_callable_fails():

    scheduler = Scheduler()

    def _boom():
        raise RuntimeError("job blew up")

    scheduler.register(job_id="backup", name="Backup", interval_seconds=300, callable_=_boom)
    client = FakeSQSClient(messages=[_message("r1", "backup")])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    worker.poll_once()

    assert client.deleted_receipt_handles == ["r1"]
    assert scheduler.history("backup")[-1].success is False


def test_poll_once_deletes_message_for_an_unregistered_job_id():

    scheduler = Scheduler()
    client = FakeSQSClient(messages=[_message("r1", "does-not-exist")])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    processed = worker.poll_once()

    assert processed == 1
    assert client.deleted_receipt_handles == ["r1"]


def test_poll_once_with_no_messages_returns_zero():

    scheduler = Scheduler()
    client = FakeSQSClient(messages=[])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    assert worker.poll_once() == 0


def test_poll_once_processes_multiple_messages_independently():

    scheduler = Scheduler()
    backup_calls = []
    health_calls = []
    scheduler.register(
        job_id="backup", name="Backup", interval_seconds=300,
        callable_=lambda: backup_calls.append(1)
    )
    scheduler.register(
        job_id="health_check", name="Health check", interval_seconds=300,
        callable_=lambda: health_calls.append(1)
    )
    client = FakeSQSClient(messages=[_message("r1", "backup"), _message("r2", "health_check")])
    worker = SQSSchedulerWorker(sqs_client=client, queue_url="q", scheduler=scheduler)

    processed = worker.poll_once()

    assert processed == 2
    assert backup_calls == [1]
    assert health_calls == [1]
    assert set(client.deleted_receipt_handles) == {"r1", "r2"}
