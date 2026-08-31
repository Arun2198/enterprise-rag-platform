from rag.generation.fallback_answerer import FallbackAnswerer


class _StubAnswerer:

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def answer(self, query, retrieved_chunks, history=None):
        self.calls.append((query, retrieved_chunks))

        if self.error is not None:
            raise self.error

        return self.result


def test_returns_primary_answer_when_primary_succeeds():

    primary = _StubAnswerer(result="primary answer")
    fallback = _StubAnswerer(result="fallback answer")
    answerer = FallbackAnswerer(primary=primary, fallback=fallback)

    result = answerer.answer("q", [])

    assert result == "primary answer"
    assert fallback.calls == []


def test_falls_back_when_primary_raises():

    primary = _StubAnswerer(error=RuntimeError("boom"))
    fallback = _StubAnswerer(result="fallback answer")
    answerer = FallbackAnswerer(primary=primary, fallback=fallback)

    result = answerer.answer("q", [])

    assert result == "fallback answer"
    assert len(fallback.calls) == 1


def test_falls_back_on_any_exception_type():

    primary = _StubAnswerer(error=ValueError("access denied due to payment instrument"))
    fallback = _StubAnswerer(result="fallback answer")
    answerer = FallbackAnswerer(primary=primary, fallback=fallback)

    assert answerer.answer("q", []) == "fallback answer"


def test_passes_query_and_chunks_through_unchanged():

    primary = _StubAnswerer(result="ok")
    fallback = _StubAnswerer(result="ok")
    answerer = FallbackAnswerer(primary=primary, fallback=fallback)

    answerer.answer("what is x", ["chunk-placeholder"])

    assert primary.calls == [("what is x", ["chunk-placeholder"])]


def test_fallback_failure_propagates_when_primary_also_fails():

    primary = _StubAnswerer(error=RuntimeError("primary down"))
    fallback = _StubAnswerer(error=RuntimeError("fallback down"))
    answerer = FallbackAnswerer(primary=primary, fallback=fallback)

    try:
        answerer.answer("q", [])
        raise AssertionError("expected the fallback's exception to propagate")
    except RuntimeError as ex:
        assert str(ex) == "fallback down"
