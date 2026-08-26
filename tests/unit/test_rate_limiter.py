from app.rate_limiter import InMemoryRateLimiter


def test_allows_requests_up_to_the_limit():

    limiter = InMemoryRateLimiter(requests_per_window=3, window_seconds=60.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True


def test_rejects_requests_beyond_the_limit_within_the_same_window():

    limiter = InMemoryRateLimiter(requests_per_window=2, window_seconds=60.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_tracks_separate_windows_per_key():

    limiter = InMemoryRateLimiter(requests_per_window=1, window_seconds=60.0)

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False
    assert limiter.allow("client-b") is False


def test_resets_after_the_window_elapses(monkeypatch):

    limiter = InMemoryRateLimiter(requests_per_window=1, window_seconds=10.0)
    fake_now = [1000.0]
    monkeypatch.setattr("app.rate_limiter.time.monotonic", lambda: fake_now[0])

    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False

    fake_now[0] += 10.0

    assert limiter.allow("client-a") is True
