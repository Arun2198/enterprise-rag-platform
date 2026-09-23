from rag.guardrails.groundedness import blended_groundedness_score
from rag.guardrails.groundedness import token_overlap


class _StubEmbedder:
    """Deterministic 2D embedder: text containing MARKER embeds to
    [1, 0], everything else to [0, 1] - exact cosine similarity of
    1.0 or 0.0 without depending on a real model."""

    def embed(self, text):
        return [1.0, 0.0] if "MARKER" in text else [0.0, 1.0]


def test_token_overlap_is_the_fraction_of_text_a_tokens_found_in_text_b():

    score = token_overlap("the cat sat on the mat", "the mat is red")

    # distinct tokens in text_a: {the, cat, sat, on, mat} = 5
    # found in text_b: {the, mat} = 2
    assert score == 2 / 5


def test_token_overlap_is_zero_for_completely_disjoint_text():

    assert token_overlap("apples and oranges", "completely unrelated words") == 0.0


def test_token_overlap_is_zero_when_either_text_is_empty():

    assert token_overlap("", "some text") == 0.0
    assert token_overlap("some text", "") == 0.0


def test_blended_score_falls_back_to_token_overlap_without_an_embedder():

    score = blended_groundedness_score("the cat sat", "the cat sat here")

    assert score == token_overlap("the cat sat", "the cat sat here")


def test_blended_score_combines_token_overlap_and_embedding_similarity():

    embedder = _StubEmbedder()

    # "foo"/"bar" share no tokens, but both texts contain MARKER so the
    # embedder scores them as identical (cosine similarity 1.0) - only
    # the shared "marker" token itself contributes to overlap.
    score = blended_groundedness_score(
        "MARKER foo", "MARKER bar", embedder=embedder,
        token_overlap_weight=0.6, similarity_weight=0.4
    )

    # token_overlap({"marker", "foo"}, {"marker", "bar"}) = 1/2 = 0.5
    assert score == 0.6 * 0.5 + 0.4 * 1.0


def test_blended_score_is_symmetric_in_what_the_two_texts_represent():
    """
    The whole point of extracting this out of HallucinationDetector -
    the math doesn't care whether it's scoring (answer, chunk) or
    (query, chunk); it's a generic pairwise grounding score.
    """
    embedder = _StubEmbedder()

    forward = blended_groundedness_score("MARKER text one", "some MARKER context", embedder=embedder)
    same_pair_different_role = blended_groundedness_score(
        "some MARKER context", "MARKER text one", embedder=embedder
    )

    # not asserting equality (token_overlap itself is directional), just
    # that both call the same underlying computation without error and
    # produce a valid bounded score
    assert 0.0 <= forward <= 1.0
    assert 0.0 <= same_pair_different_role <= 1.0
