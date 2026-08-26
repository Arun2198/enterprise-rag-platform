# Chapter 12: Testing Strategy

`tests/unit/` contains **413 tests** across 61 files (verified by running `uv run pytest
--collect-only` against the current codebase), structured to mirror `src/` one file at a time —
`test_recursive_chunker.py` tests `rag/chunking/recursive_chunker.py`, `test_pii_guard.py` tests
`rag/guardrails/pii_guard.py`, and so on. This chapter covers how the suite is built, and
specifically how it manages to be fast and fully offline despite the app depending on real,
normally-slow ML models.

## 1. Why "no network calls in tests" is a real design goal, not an afterthought

Two of this project's real dependencies are large neural network models: the sentence-transformer
embedding model ([Chapter 3](03-embeddings-and-vector-search.md)) and the cross-encoder reranker
([Chapter 4](04-retrieval-and-reranking.md)). Both are:

- **Slow to load** — loading a real transformer model takes real seconds, and doing that once per
  test (or even once per test *file*) across hundreds of tests would make the suite unusably slow
  to run repeatedly during development.
- **Network-dependent on first use** — `sentence-transformers` downloads a model from HuggingFace
  the first time it's requested, meaning a test run without internet access, or in a sandboxed CI
  runner with no outbound network, would simply fail.
- **Non-deterministic in ways irrelevant to what's being tested** — a test asserting "chunk A
  should rank above chunk B" cares about *relative ranking behavior*, not about the literal
  numeric output of a specific pretrained model.

None of this means the real models go untested — [`test_sentence_transformer_embedder.py`](
../../tests/unit/test_sentence_transformer_embedder.py) exists specifically to exercise the real
class (see section 3). It means the *rest* of the suite — the hundreds of tests that need *some*
embedder or reranker to exist so `RAGService` can be constructed, but don't care which one — get a
fast, deterministic, offline stand-in instead.

## 2. The global fake pattern: `tests/unit/conftest.py`

Pytest automatically loads `conftest.py` before running any test in its directory. This project's
`tests/unit/conftest.py` uses that to install three permanent substitutions **once, at test-session
start**, for the entire run:

```python
patch("rag.retrieval.reranker.CrossEncoder", _FakeCrossEncoder).start()
patch("rag.guardrails.nli_hallucination_detector.CrossEncoder", _FakeNLICrossEncoder).start()
patch("rag.embeddings.sentence_transformer_embedder.SentenceTransformer", _FakeSentenceTransformer).start()
```

Note `.start()` with no matching `.stop()` anywhere — this is deliberate. A normal `unittest.mock
.patch` is usually used as a context manager or decorator, active only for the duration of one
test, then automatically reverted. Here, the intent is the opposite: replace these classes for the
*entire* test process, no matter which test file or code path ends up importing them, so nothing
in the whole suite can ever accidentally trigger a real download.

### `_FakeSentenceTransformer` — a fake that still behaves like a real embedder

```python
class _FakeSentenceTransformer:
    def __init__(self, model_name=None, *args, **kwargs) -> None:
        self._hashing = HashingEmbedder(dimensions=384)

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, text, normalize_embeddings=True, **kwargs) -> _FakeEmbeddingTensor:
        return _FakeEmbeddingTensor(self._hashing.embed(text))
```

Rather than returning a fixed, meaningless value for every input (which would make any test
asserting "chunk A should outrank chunk B for this query" impossible to write correctly), it
delegates internally to the real `HashingEmbedder` ([Chapter 3](
03-embeddings-and-vector-search.md#3-this-projects-two-embedders)) — so tests still get vectors
that are genuinely sensitive to word content and can be meaningfully compared for ranking
purposes, just without any neural network or network call involved. `_FakeEmbeddingTensor` exists
purely because real `SentenceTransformer.encode()` returns a numpy array with a `.tolist()`
method, which `SentenceTransformerEmbedder.embed()` calls — the fake has to expose that same shape
to be a drop-in substitute.

`_FakeCrossEncoder` and `_FakeNLICrossEncoder` follow the same principle for the reranker and NLI
hallucination detector, each shaped to match what their respective real class's `.predict()`
method is expected to return (a flat list of scores for the reranker; a `(n, 3)`
contradiction/entailment/neutral probability shape for NLI).

**The practical effect**: `app.main`'s module-level `build_rag_service()` call — the exact same
factory function ([Chapter 1](01-project-overview.md)) the live deployed app uses — runs during
test collection and never touches the network, because by the time any test file imports
`app.main`, `conftest.py` has already substituted the underlying model classes.

## 3. Testing the real thing anyway

`test_sentence_transformer_embedder.py` and the reranker's equivalent tests exist specifically to
verify the *real* wrapper classes (`SentenceTransformerEmbedder`, `CrossEncoderReranker`) behave
correctly — they patch over the session-wide fake with their own more targeted mock for the
duration of just that test, verifying things like "the class correctly calls `.tolist()` on
whatever `encode()` returns" without needing an actual model download to prove it. This is the
general shape used throughout: the global fake covers "everything that just needs *an* embedder to
exist," and individual test files cover "does this specific wrapper class use its dependency
correctly."

## 4. Shared OpenTelemetry setup

```python
TELEMETRY_READER = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[TELEMETRY_READER]))
```

OpenTelemetry only allows a `MeterProvider` to be set **once** per process — a second call is
silently ignored, not an error, which would make it impossible for later tests to see their own
independent provider if each tried to configure one. `conftest.py` sets exactly one shared
provider with an in-memory reader for the whole session; `test_guardrails_telemetry.py` and
`test_mlops_telemetry.py` ([Chapters 6](06-guardrails-and-safety.md) and
[8](08-mlops-platform.md)) read from that same shared reader. Because metric streams accumulate
for the rest of the session once any test emits one, those tests use a **delta-based** assertion
pattern — checking the *change* in a counter's value across a specific action, not its absolute
value — since the absolute value depends on everything that ran before it in the same session.

## 5. `RAGService()` direct construction vs. `build_rag_service()` in tests

Most unit tests construct `RAGService()` with no arguments — this gets the fast, fully offline
defaults from [Chapter 1](
01-project-overview.md#4-the-core-architectural-idea-provider-swap-via-protocols):
`HashingEmbedder`, `InMemoryVectorStore`, `ExtractiveAnswerer`, no reranker, no guardrails, no
ingest-path restriction. Tests that specifically need to verify factory wiring
(`test_service_factory.py`) or end-to-end API behavior through the real app
(`test_api.py`) go through `build_rag_service()`/the real `app.main` module instead — exercising
the actual production wiring path, just with the model classes faked out underneath it by
`conftest.py`, not with the wiring itself mocked.

## 6. `monkeypatch` for testing module-level state

`app/main.py` builds `rag_service` and `platform_manager` once, at import time, as module-level
variables ([Chapter 8](08-mlops-platform.md), [Chapter 13](13-security-and-glossary.md)). To test
what happens when startup *fails* (e.g. `/health` should return `503` with the real error) without
actually having to trigger a real failing import, `test_api.py` uses pytest's `monkeypatch` fixture
to directly overwrite those module attributes for the duration of one test:

```python
def test_health_reports_503_when_startup_failed(monkeypatch):
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]
```

`monkeypatch`, unlike the session-wide `unittest.mock.patch(...).start()` pattern in `conftest.py`,
automatically reverts its change after the test function returns — appropriate here because this
particular substitution is meant to be scoped to one test, not the whole session.

## 7. `tests/unit/test_api.py` — the one true end-to-end test

Every other test file is a unit test for one module in isolation. `test_api.py` is different: it
drives the real FastAPI application through `fastapi.testclient.TestClient`, which means a request
genuinely flows through every layer — HTTP routing, request validation, `RAGService.ingest()`/
`.ask()`, the real (fake-model-backed) retrieval and generation pipeline, the real guardrails
manager — exactly as [Chapter 5](
05-generation-and-llms.md#6-tracing-one-real-question-through-generation)'s worked example walks
through, verified as a real assertion (`test_ingest_and_ask_endpoints`) rather than just described
in prose.

Next: [Chapter 13 — Security, Known Gaps, and Glossary](13-security-and-glossary.md).
