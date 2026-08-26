import time
from threading import Lock


class InMemoryRateLimiter:
    """
    Fixed-window, in-process rate limiter - no external store (Redis,
    DynamoDB) needed at this project's traffic scale, same
    dependency-light philosophy as the rest of this codebase's
    hand-rolled pieces (custom BM25, custom SigV4 client).

    Real limitation, stated plainly rather than hidden: this only
    coordinates within a single process. With N ECS tasks running this
    app, each task enforces its own independent window against its own
    in-memory state, so the *effective* global limit across the whole
    service is up to N times requests_per_window - the same category of
    caveat the pre-EventBridge interval scheduler had (see
    mlops/sqs_scheduler_worker.py). Correct and sufficient at the
    single-task deployment this repo has actually run against AWS; a
    deployment that needs an exact global limit across multiple tasks
    would need a shared store instead.
    """

    def __init__(
        self,
        requests_per_window: int,
        window_seconds: float
    ) -> None:
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._lock = Lock()
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(
        self,
        key: str
    ) -> bool:
        now = time.monotonic()

        with self._lock:
            window_start, count = self._windows.get(key, (now, 0))

            if now - window_start >= self.window_seconds:
                window_start, count = now, 0

            count += 1
            self._windows[key] = (window_start, count)
            return count <= self.requests_per_window
