import logging
import time
from dataclasses import replace
from datetime import datetime
from datetime import timezone
from typing import Callable

from mlops.schemas import JobRun
from mlops.schemas import ScheduledJob

logger = logging.getLogger(__name__)


class JobNotFoundError(KeyError):
    pass


class DuplicateJobError(ValueError):
    pass


class Scheduler:
    """
    Real, working job registry with interval-based due-job execution.
    Deliberately owns no background thread or daemon - call
    run_due_jobs() periodically from whatever actually owns scheduling
    in a given deployment (a simple loop, a Kubernetes CronJob, a GitHub
    Actions schedule trigger, plain cron). Keeps this dependency-free
    and trivially testable with a fake clock, instead of needing real
    sleeping/threading in tests.

    Example jobs: re-index documents, run evaluation, execute
    experiments, health checks, drift detection, backup - register any
    of these as a zero-argument callable.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._callables: dict[str, Callable[[], None]] = {}
        self._runs: dict[str, list[JobRun]] = {}

    def register(
        self,
        job_id: str,
        name: str,
        interval_seconds: float,
        callable_: Callable[[], None],
        start_at: float | None = None
    ) -> ScheduledJob:
        if job_id in self._jobs:
            raise DuplicateJobError(f"job already registered: {job_id}")

        job = ScheduledJob(
            job_id=job_id,
            name=name,
            interval_seconds=interval_seconds,
            next_run_at=start_at if start_at is not None else time.time()
        )
        self._jobs[job_id] = job
        self._callables[job_id] = callable_
        self._runs[job_id] = []
        logger.info(
            "job_registered",
            extra={"job_id": job_id, "interval_seconds": interval_seconds}
        )
        return job

    def enable(
        self,
        job_id: str
    ) -> ScheduledJob:
        return self._set_enabled(job_id, True)

    def disable(
        self,
        job_id: str
    ) -> ScheduledJob:
        return self._set_enabled(job_id, False)

    def trigger(
        self,
        job_id: str
    ) -> JobRun:
        """Run a job immediately, outside its schedule, and reschedule it from now."""
        return self._execute(job_id, now=time.time())

    def run_due_jobs(
        self,
        now: float | None = None
    ) -> list[JobRun]:
        now = now if now is not None else time.time()
        runs = []

        for job_id, job in list(self._jobs.items()):
            if job.enabled and job.next_run_at <= now:
                runs.append(self._execute(job_id, now=now))

        return runs

    def history(
        self,
        job_id: str
    ) -> list[JobRun]:
        if job_id not in self._runs:
            raise JobNotFoundError(job_id)

        return list(self._runs[job_id])

    def list_jobs(self) -> list[ScheduledJob]:
        return list(self._jobs.values())

    def _execute(
        self,
        job_id: str,
        now: float
    ) -> JobRun:
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)

        job = self._jobs[job_id]
        started_at = _now_iso()
        error = None
        success = True

        try:
            self._callables[job_id]()
        except Exception as ex:
            success = False
            error = str(ex)
            logger.warning("job_failed", extra={"job_id": job_id, "error": str(ex)})

        run = JobRun(
            job_id=job_id,
            started_at=started_at,
            finished_at=_now_iso(),
            success=success,
            error=error
        )
        self._runs[job_id].append(run)
        self._jobs[job_id] = replace(job, next_run_at=now + job.interval_seconds)
        logger.info("job_executed", extra={"job_id": job_id, "success": success})
        return run

    def _set_enabled(
        self,
        job_id: str,
        enabled: bool
    ) -> ScheduledJob:
        if job_id not in self._jobs:
            raise JobNotFoundError(job_id)

        updated = replace(self._jobs[job_id], enabled=enabled)
        self._jobs[job_id] = updated
        return updated


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
