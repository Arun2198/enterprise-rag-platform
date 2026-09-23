import re

from rag.embeddings.base import Embedder


def token_overlap(
    text_a: str,
    text_b: str
) -> float:
    """Fraction of text_a's distinct tokens that also appear in text_b."""
    terms_a = _tokens(text_a)

    if not terms_a:
        return 0.0

    terms_b = _tokens(text_b)

    if not terms_b:
        return 0.0

    return len(terms_a.intersection(terms_b)) / len(terms_a)


def blended_groundedness_score(
    text_a: str,
    text_b: str,
    embedder: Embedder | None = None,
    token_overlap_weight: float = 0.6,
    similarity_weight: float = 0.4
) -> float:
    """
    How much text_a is "grounded in" text_b: token overlap, blended with
    embedding cosine similarity when an embedder is available. The
    underlying math doesn't care what the two texts represent - shared
    by HallucinationDetector (answer vs. retrieved chunk, checking
    whether a generated answer is actually grounded) and RAGService's
    grounded-first routing (query vs. retrieved chunk, checking whether
    retrieval alone already covers the question well enough to skip an
    LLM call) - same scoring logic, different pair of texts.
    """
    overlap = token_overlap(text_a, text_b)

    if embedder is None or not text_a.strip() or not text_b.strip():
        return overlap

    similarity = _cosine_similarity(
        embedder.embed(text_a),
        embedder.embed(text_b)
    )
    blended = token_overlap_weight * overlap + similarity_weight * similarity
    return max(0.0, min(blended, 1.0))


def _tokens(
    text: str
) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _cosine_similarity(
    first: list[float],
    second: list[float]
) -> float:
    numerator = sum(a * b for a, b in zip(first, second, strict=True))
    first_norm = sum(a * a for a in first) ** 0.5
    second_norm = sum(b * b for b in second) ** 0.5

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return numerator / (first_norm * second_norm)
