import math

from rag.embeddings.hashing_embedder import HashingEmbedder


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def test_embed_is_deterministic_for_the_same_text():

    embedder = HashingEmbedder()

    first = embedder.embed("the quick brown fox")
    second = embedder.embed("the quick brown fox")

    assert first == second


def test_embed_returns_the_configured_dimensionality():

    embedder = HashingEmbedder(dimensions=128)

    vector = embedder.embed("some text")

    assert len(vector) == 128


def test_embed_defaults_to_384_dimensions():

    embedder = HashingEmbedder()

    assert len(embedder.embed("text")) == 384


def test_embed_returns_a_unit_vector_for_non_empty_text():

    embedder = HashingEmbedder()

    vector = embedder.embed("some reasonably long piece of text to embed")
    norm = math.sqrt(sum(v * v for v in vector))

    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_embed_returns_zero_vector_for_empty_text():

    embedder = HashingEmbedder(dimensions=16)

    vector = embedder.embed("")

    assert vector == [0.0] * 16


def test_embed_returns_zero_vector_for_text_with_no_alphanumeric_tokens():

    embedder = HashingEmbedder(dimensions=16)

    vector = embedder.embed("!!! --- ...")

    assert vector == [0.0] * 16


def test_embed_is_case_insensitive():

    embedder = HashingEmbedder()

    assert embedder.embed("Contractors Receive Leave") == embedder.embed("contractors receive leave")


def test_embed_ignores_punctuation_between_identical_tokens():

    embedder = HashingEmbedder()

    assert embedder.embed("leave, policy!") == embedder.embed("leave policy")


def test_shared_vocabulary_scores_higher_similarity_than_unrelated_text():

    embedder = HashingEmbedder()

    query = embedder.embed("contractors receive leave days")
    related = embedder.embed("contractors are entitled to ten days of leave")
    unrelated = embedder.embed("the quarterly revenue report shows growth")

    assert _cosine(query, related) > _cosine(query, unrelated)


def test_identical_text_has_cosine_similarity_of_one():

    embedder = HashingEmbedder()
    vector = embedder.embed("employees receive twenty days of paid leave")

    assert math.isclose(_cosine(vector, vector), 1.0, abs_tol=1e-9)
