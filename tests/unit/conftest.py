from unittest.mock import patch

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from rag.embeddings.hashing_embedder import HashingEmbedder

# OTel only allows a MeterProvider to be set once per process, so every
# telemetry test in the suite (guardrails, mlops, ...) shares this one
# reader instead of each trying to set its own - see
# test_guardrails_telemetry.py / test_mlops_telemetry.py for the
# delta-based assertion pattern this requires (metric streams persist
# and accumulate for the rest of the session once any test emits one).
TELEMETRY_READER = InMemoryMetricReader()
metrics.set_meter_provider(MeterProvider(metric_readers=[TELEMETRY_READER]))


class _FakeCrossEncoder:
    """
    Session-wide stand-in for sentence_transformers.CrossEncoder. Reranking
    is enabled by default (RERANKER_ENABLED defaults to true), so anything
    that builds a RAGService through the normal wiring - including
    app.main's module-level build_rag_service() call - would otherwise
    download a real model. Individual tests that care about specific
    CrossEncoder behavior patch over this with their own mock.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, pairs, *args, **kwargs):
        return [0.0 for _ in pairs]


class _FakeNLICrossEncoder:
    """
    Same idea as _FakeCrossEncoder but shaped for NLIHallucinationDetector,
    which calls predict(pairs, apply_softmax=True) expecting a
    (n_pairs, 3) array of [contradiction, entailment, neutral]
    probabilities. Not registered by GuardrailManager.default(), but
    nothing should download a real model just by importing the module.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, pairs, *args, **kwargs):
        return [[0.0, 0.0, 1.0] for _ in pairs]


class _FakeEmbeddingTensor:
    """Minimal stand-in for the numpy array sentence-transformers returns -
    SentenceTransformerEmbedder.embed() calls .tolist() on it."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeSentenceTransformer:
    """
    Session-wide stand-in for sentence_transformers.SentenceTransformer.
    EMBEDDING_PROVIDER defaults to sentence_transformer (see
    app/config.py - the hashing embedder is explicitly not good enough to
    be the project's default), so anything that builds a RAGService
    through the normal wiring - including app.main's module-level
    build_rag_service() call - would otherwise download a real model on
    every test run. Delegates to HashingEmbedder internally so retrieval
    ranking in tests stays deterministic and content-sensitive (a real
    embedding's job) without any network dependency; tests that care about
    actually exercising a real sentence-transformers model patch this over
    themselves (see test_sentence_transformer_embedder.py).
    """

    def __init__(self, model_name: str | None = None, *args, **kwargs) -> None:
        self._hashing = HashingEmbedder(dimensions=384)

    def get_sentence_embedding_dimension(self) -> int:
        return 384

    def encode(self, text, normalize_embeddings: bool = True, **kwargs) -> _FakeEmbeddingTensor:
        return _FakeEmbeddingTensor(self._hashing.embed(text))


patch("rag.retrieval.reranker.CrossEncoder", _FakeCrossEncoder).start()
patch(
    "rag.guardrails.nli_hallucination_detector.CrossEncoder",
    _FakeNLICrossEncoder
).start()
patch(
    "rag.embeddings.sentence_transformer_embedder.SentenceTransformer",
    _FakeSentenceTransformer
).start()
