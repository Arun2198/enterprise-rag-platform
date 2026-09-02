from rag.chunking.chunk import Chunk
from rag.generation.document_first_answerer import DocumentFirstAnswerer
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class _MarkerEmbedder:
    """Deterministic 2D embedder: text containing MARKER embeds to
    [1, 0], everything else to [0, 1] - gives an exact, non-flaky
    cosine similarity of 1.0 (relevant) or 0.0 (irrelevant)."""
    dimensions = 2
    provider_name = "test_dense"
    model_name = "marker-embedder"

    def embed(self, text):
        return [1.0, 0.0] if "MARKER" in text else [0.0, 1.0]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class _RecordingAnswerer:

    def __init__(self, label):
        self.label = label
        self.calls = 0

    def answer(self, query, retrieved_chunks, history=None):
        self.calls += 1
        return f"{self.label} answer"


def _retrieved_chunk(text):
    chunk = Chunk(
        chunk_id="doc:0", document_id="doc", chunk_index=0,
        text=text, source="doc.md", document_type="markdown"
    )
    return RetrievedChunk(chunk=chunk, vector_score=1.0, keyword_score=1.0, score=1.0)


def test_routes_to_document_answerer_when_relevance_meets_threshold():
    document_answerer = _RecordingAnswerer("document")
    llm_answerer = _RecordingAnswerer("llm")
    answerer = DocumentFirstAnswerer(
        document_answerer=document_answerer,
        llm_answerer=llm_answerer,
        embedder=_MarkerEmbedder(),
        threshold=0.5
    )

    result = answerer.answer("MARKER query", [_retrieved_chunk("MARKER matching content")])

    assert result == "document answer"
    assert document_answerer.calls == 1
    assert llm_answerer.calls == 0


def test_routes_to_llm_answerer_when_relevance_below_threshold():
    document_answerer = _RecordingAnswerer("document")
    llm_answerer = _RecordingAnswerer("llm")
    answerer = DocumentFirstAnswerer(
        document_answerer=document_answerer,
        llm_answerer=llm_answerer,
        embedder=_MarkerEmbedder(),
        threshold=0.5
    )

    result = answerer.answer("completely unrelated query", [_retrieved_chunk("MARKER matching content")])

    assert result == "llm answer"
    assert document_answerer.calls == 0
    assert llm_answerer.calls == 1


def test_routes_to_llm_answerer_when_nothing_was_retrieved():
    document_answerer = _RecordingAnswerer("document")
    llm_answerer = _RecordingAnswerer("llm")
    answerer = DocumentFirstAnswerer(
        document_answerer=document_answerer,
        llm_answerer=llm_answerer,
        embedder=_MarkerEmbedder(),
        threshold=0.5
    )

    result = answerer.answer("anything", [])

    assert result == "llm answer"
    assert document_answerer.calls == 0
    assert llm_answerer.calls == 1


def test_default_threshold_comes_from_the_embedder_appropriate_default():
    from rag.guardrails.retrieval_relevance_guard import DENSE_EMBEDDER_DEFAULT_THRESHOLD

    answerer = DocumentFirstAnswerer(
        document_answerer=_RecordingAnswerer("document"),
        llm_answerer=_RecordingAnswerer("llm"),
        embedder=_MarkerEmbedder()
    )

    assert answerer.threshold == DENSE_EMBEDDER_DEFAULT_THRESHOLD


def test_explicit_threshold_overrides_the_default():
    answerer = DocumentFirstAnswerer(
        document_answerer=_RecordingAnswerer("document"),
        llm_answerer=_RecordingAnswerer("llm"),
        embedder=_MarkerEmbedder(),
        threshold=0.9
    )

    assert answerer.threshold == 0.9
