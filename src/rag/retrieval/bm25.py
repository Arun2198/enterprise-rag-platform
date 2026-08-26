import math
import re


def tokenize(
    text: str
) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_bm25(
    query_terms: list[str],
    documents: list[tuple[str, list[str]]],
    k1: float = 1.2,
    b: float = 0.75
) -> dict[str, float]:
    """
    Real Okapi BM25 (with the standard +0.5 IDF smoothing, k1=1.2/b=0.75
    defaults) - term frequency, inverse document frequency, and document
    length normalization, not a word-overlap approximation. Used by
    InMemoryVectorStore.search_lexical() so local/test retrieval scores
    lexical relevance the same way the OpenSearch-backed path does, instead
    of two different algorithms silently diverging between dev and prod.

    documents: list of (id, tokens) pairs already tokenized by the caller.
    Returns {id: score} for documents that matched at least one query term -
    documents with zero overlap are omitted, not scored at 0.0.
    """
    if not documents or not query_terms:
        return {}

    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in documents}
    avg_doc_length = sum(doc_lengths.values()) / len(documents)
    document_count = len(documents)

    document_frequency = {
        term: sum(1 for _, tokens in documents if term in tokens)
        for term in set(query_terms)
    }
    inverse_document_frequency = {
        term: math.log(1 + (document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
        for term in document_frequency
    }

    scores: dict[str, float] = {}

    for doc_id, tokens in documents:
        term_frequency: dict[str, int] = {}

        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        doc_length = doc_lengths[doc_id]
        score = 0.0

        for term in query_terms:
            frequency = term_frequency.get(term)

            if not frequency:
                continue

            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (1 - b + b * doc_length / avg_doc_length)
            score += inverse_document_frequency[term] * numerator / denominator

        if score > 0:
            scores[doc_id] = score

    return scores
