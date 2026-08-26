from rag.chunking.chunk import Chunk
from rag.guardrails.base import GuardrailContext
from rag.guardrails.retrieval_relevance_guard import DENSE_EMBEDDER_DEFAULT_THRESHOLD
from rag.guardrails.retrieval_relevance_guard import HASHING_EMBEDDER_DEFAULT_THRESHOLD
from rag.guardrails.retrieval_relevance_guard import RetrievalRelevanceGuard
from rag.guardrails.retrieval_relevance_guard import default_retrieval_relevance_threshold
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class _FixedEmbedder:
    """
    Returns whatever vector was pre-registered for a given text, so a test
    can pin exact cosine similarities instead of depending on a real
    embedding model's actual output.
    """

    def __init__(self, vectors: dict[str, list[float]], provider_name: str = "test"):
        self._vectors = vectors
        self.provider_name = provider_name
        self.dimensions = len(next(iter(vectors.values())))

    def embed(self, text):
        return self._vectors[text]

    def embed_batch(self, texts):
        return [self._vectors[t] for t in texts]


def _mk_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="d1", chunk_index=0,
        text=text, source="test", document_type="md"
    )


def _mk_retrieved(text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_mk_chunk("d1:0", text),
        vector_score=0.9, keyword_score=0.0, score=0.9
    )


def test_triggers_when_best_similarity_is_below_threshold():

    embedder = _FixedEmbedder({
        "query": [1.0, 0.0],
        "chunk": [0.0, 1.0],  # orthogonal - cosine similarity 0.0
    })
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.5)
    context = GuardrailContext(query="query", answer="x", retrieved_chunks=[_mk_retrieved("chunk")])

    finding = guard.check(context)

    assert finding.triggered is True
    assert finding.metadata["low_retrieval_relevance"] is True
    assert finding.metadata["retrieval_relevance_score"] == 0.0


def test_does_not_trigger_when_best_similarity_meets_threshold():

    embedder = _FixedEmbedder({
        "query": [1.0, 0.0],
        "chunk": [1.0, 0.0],  # identical - cosine similarity 1.0
    })
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.5)
    context = GuardrailContext(query="query", answer="x", retrieved_chunks=[_mk_retrieved("chunk")])

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.metadata["retrieval_relevance_score"] == 1.0


def test_uses_the_best_of_up_to_three_retrieved_chunks():

    embedder = _FixedEmbedder({
        "query": [1.0, 0.0],
        "poor_match": [0.0, 1.0],
        "good_match": [1.0, 0.0],
    })
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.5)
    context = GuardrailContext(
        query="query", answer="x",
        retrieved_chunks=[_mk_retrieved("poor_match"), _mk_retrieved("good_match")]
    )

    finding = guard.check(context)

    assert finding.triggered is False
    assert finding.metadata["retrieval_relevance_score"] == 1.0


def test_skips_when_no_embedder():

    guard = RetrievalRelevanceGuard(embedder=None, threshold=0.5)
    context = GuardrailContext(query="query", answer="x", retrieved_chunks=[_mk_retrieved("chunk")])

    finding = guard.check(context)

    assert finding.triggered is False


def test_skips_when_no_retrieved_chunks():

    embedder = _FixedEmbedder({"query": [1.0, 0.0]})
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.5)
    context = GuardrailContext(query="query", answer="x", retrieved_chunks=[])

    finding = guard.check(context)

    assert finding.triggered is False


def test_skips_on_empty_query():

    embedder = _FixedEmbedder({"chunk": [1.0, 0.0]})
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.5)
    context = GuardrailContext(query="", answer="x", retrieved_chunks=[_mk_retrieved("chunk")])

    finding = guard.check(context)

    assert finding.triggered is False


def test_default_threshold_is_lower_for_hashing_embedder():

    class _HashingLike:
        provider_name = "hashing"

    class _DenseLike:
        provider_name = "sentence_transformer"

    assert default_retrieval_relevance_threshold(_HashingLike()) == HASHING_EMBEDDER_DEFAULT_THRESHOLD
    assert default_retrieval_relevance_threshold(_DenseLike()) == DENSE_EMBEDDER_DEFAULT_THRESHOLD
    assert HASHING_EMBEDDER_DEFAULT_THRESHOLD < DENSE_EMBEDDER_DEFAULT_THRESHOLD
    assert default_retrieval_relevance_threshold(None) == DENSE_EMBEDDER_DEFAULT_THRESHOLD


def test_explicit_threshold_overrides_the_embedder_based_default():

    embedder = _FixedEmbedder({"query": [1.0, 0.0], "chunk": [1.0, 0.0]}, provider_name="hashing")
    guard = RetrievalRelevanceGuard(embedder=embedder, threshold=0.99)

    assert guard.threshold == 0.99
