import pytest

from mlops.scheduler import DuplicateJobError
from mlops.scheduler import JobNotFoundError
from mlops.scheduler import Scheduler


def test_run_due_jobs_only_runs_jobs_that_are_due():

    scheduler = Scheduler()
    calls = []
    scheduler.register("job-a", "Job A", interval_seconds=60, callable_=lambda: calls.append("a"), start_at=100.0)
    scheduler.register("job-b", "Job B", interval_seconds=60, callable_=lambda: calls.append("b"), start_at=200.0)

    runs = scheduler.run_due_jobs(now=150.0)

    assert calls == ["a"]
    assert len(runs) == 1
    assert runs[0].job_id == "job-a"


def test_run_due_jobs_reschedules_next_run_at():

    scheduler = Scheduler()
    scheduler.register("job-a", "Job A", interval_seconds=60, callable_=lambda: None, start_at=100.0)

    scheduler.run_due_jobs(now=100.0)

    job = scheduler.list_jobs()[0]
    assert job.next_run_at == 160.0


def test_disabled_job_is_never_run_due():

    scheduler = Scheduler()
    calls = []
    scheduler.register("job-a", "Job A", interval_seconds=60, callable_=lambda: calls.append("a"), start_at=100.0)
    scheduler.disable("job-a")

    scheduler.run_due_jobs(now=200.0)

    assert calls == []


def test_trigger_runs_immediately_outside_schedule():

    scheduler = Scheduler()
    calls = []
    scheduler.register("job-a", "Job A", interval_seconds=3600, callable_=lambda: calls.append("a"), start_at=999999.0)

    run = scheduler.trigger("job-a")

    assert calls == ["a"]
    assert run.success is True


def test_failing_job_records_failure_without_crashing_scheduler():

    scheduler = Scheduler()

    def failing():
        raise RuntimeError("boom")

    scheduler.register("job-a", "Job A", interval_seconds=60, callable_=failing, start_at=100.0)

    run = scheduler.trigger("job-a")

    assert run.success is False
    assert run.error == "boom"
    assert scheduler.history("job-a")[0].success is False


def test_duplicate_registration_raises():

    scheduler = Scheduler()
    scheduler.register("job-a", "Job A", interval_seconds=60, callable_=lambda: None)

    with pytest.raises(DuplicateJobError):
        scheduler.register("job-a", "Job A again", interval_seconds=60, callable_=lambda: None)


def test_unknown_job_operations_raise_job_not_found():

    scheduler = Scheduler()

    with pytest.raises(JobNotFoundError):
        scheduler.trigger("does-not-exist")

    with pytest.raises(JobNotFoundError):
        scheduler.history("does-not-exist")

    with pytest.raises(JobNotFoundError):
        scheduler.enable("does-not-exist")


def test_history_accumulates_across_multiple_runs():

    scheduler = Scheduler()
    scheduler.register("job-a", "Job A", interval_seconds=10, callable_=lambda: None, start_at=0.0)

    scheduler.run_due_jobs(now=0.0)
    scheduler.run_due_jobs(now=10.0)
    scheduler.run_due_jobs(now=20.0)

    assert len(scheduler.history("job-a")) == 3
