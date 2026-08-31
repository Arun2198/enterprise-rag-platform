from rag.embeddings.base import Embedder
from rag.guardrails.base import Action
from rag.guardrails.base import GuardrailContext
from rag.guardrails.base import GuardrailFinding
from rag.guardrails.base import GuardrailStage
from rag.guardrails.base import Severity

# Calibrated with scripts/retrieval_relevance_guard_verification.py: for
# each of evaluation/golden_dataset.json's 24 real queries, ran the query
# through the actual RAGService._retrieve() pipeline (not an idealized
# full-corpus scan) and recorded this guard's real best-of-top-3 score,
# against sample_documents/AI-RMF-1stdraft.pdf - then did the same for 4
# genuinely unanswerable (off-topic) queries.
#
#   BAAI/bge-small-en-v1.5 (via SentenceTransformerEmbedder):
#     answerable min 0.716 (24/24 queries, none below it)
#     unanswerable: 0.42, 0.47, 0.65, 0.73 (highest overlaps the
#     answerable range - "python loc" is a known miss, not caught)
#     -> 0.68 catches 3/4 unanswerable cases with zero false positives
#        against all 24 real answerable queries.
#
#   HashingEmbedder: answerable min 0.294, unanswerable range 0.214-0.432
#     - these ranges overlap badly (unanswerable's 0.314/0.432 sit inside
#     the answerable distribution's own worst 6 scores). There is no
#     threshold that separates them without causing false-positive
#     abstention on real, legitimate queries. HASHING_EMBEDDER_DEFAULT_
#     THRESHOLD is therefore set low enough to be a safe no-op (zero
#     false positives, but also zero real catches) rather than a
#     falsely-confident "calibrated" number - HashingEmbedder's vectors
#     just aren't good enough for this signal. This is exactly why the
#     guard defaults to disabled everywhere (GuardrailManager.default(),
#     service_factory's RETRIEVAL_RELEVANCE_GUARD_ENABLED) rather than
#     being on by default like PIIGuard/HallucinationDetector.
#
#   jina-embeddings-v3 (via JinaEmbedder, real API calls, calibrated with
#   scripts/retrieval_relevance_guard_verification_jina.py) - found the
#   hard way in the live AWS deployment: DENSE_EMBEDDER_DEFAULT_THRESHOLD
#   (0.68, calibrated for a *different* embedder) was never re-verified
#   after EMBEDDING_PROVIDER=jina became the AWS default, and in
#   production it caused false-positive abstention on a genuinely
#   on-topic, answerable query. Real measured scores: answerable range
#   0.377-0.918 (24/24 queries), unanswerable range 0.333-0.576 - these
#   overlap substantially more than bge-small-en-v1.5's did (Jina's
#   embedding space puts unrelated text closer together than bge-small's
#   does). 0.68 produces 14/24 false positives against real answerable
#   queries - unusable. 0.37 gives zero false positives (just under the
#   lowest real answerable score) and still catches 2/4 unanswerable
#   cases - same "prioritize zero false positives over catch rate"
#   choice already made for bge-small, just at a much lower absolute
#   number because Jina's similarity scale sits lower overall.
#
# Three named defaults, not one universal number - different embedding
# models produce genuinely different cosine-similarity distributions
# (embedding space anisotropy varies by model), so a threshold from one
# does not transfer to another - this was proven wrong in production
# once already for exactly this reason. Re-run the matching verification
# script and update the relevant constant whenever EMBEDDING_MODEL_NAME
# or EMBEDDING_PROVIDER changes to a materially different model/provider.
HASHING_EMBEDDER_DEFAULT_THRESHOLD = 0.20
DENSE_EMBEDDER_DEFAULT_THRESHOLD = 0.68
JINA_EMBEDDER_DEFAULT_THRESHOLD = 0.37


def default_retrieval_relevance_threshold(
    embedder: Embedder | None
) -> float:
    provider_name = getattr(embedder, "provider_name", None)

    if provider_name == "hashing":
        return HASHING_EMBEDDER_DEFAULT_THRESHOLD

    if provider_name == "jina":
        return JINA_EMBEDDER_DEFAULT_THRESHOLD

    return DENSE_EMBEDDER_DEFAULT_THRESHOLD


class RetrievalRelevanceGuard:
    """
    Groundedness (HallucinationDetector) measures whether the ANSWER
    matches the retrieved CHUNKS - it says nothing about whether those
    chunks are actually relevant to the QUERY. For ExtractiveAnswerer in
    particular, the "answer" is copied verbatim from a chunk, so
    groundedness is close to tautological: it can't catch a case where
    retrieval confidently returns topically irrelevant content for a
    query the corpus simply doesn't answer (e.g. asking an AI-policy
    document about the boiling point of mercury). This guard adds the
    missing signal directly: cosine similarity between the query and the
    best-matching retrieved chunk's text, both freshly embedded through
    the same Embedder retrieval already uses - independent of whatever
    scoring scale the underlying vector store backend reports (RRF-fused
    scores are rank-based, not magnitude-based, and raw vector store
    scores aren't comparable across backends - see the calibration note
    above for why an embedder-appropriate absolute threshold is used
    instead of either of those).
    """
    name = "retrieval_relevance_guard"
    stage = GuardrailStage.OUTPUT

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float | None = None
    ) -> None:
        self.embedder = embedder
        self.threshold = (
            threshold if threshold is not None
            else default_retrieval_relevance_threshold(embedder)
        )

    def check(
        self,
        context: GuardrailContext
    ) -> GuardrailFinding:
        if self.embedder is None or not context.retrieved_chunks:
            return GuardrailFinding(
                guardrail_name=self.name,
                triggered=False,
                severity=Severity.INFO,
                action=Action.ALLOW,
                message="no embedder or no retrieved chunks - relevance check skipped"
            )

        query = context.query or ""

        if not query.strip():
            return GuardrailFinding(
                guardrail_name=self.name,
                triggered=False,
                severity=Severity.INFO,
                action=Action.ALLOW,
                message="empty query - relevance check skipped"
            )

        query_embedding = self.embedder.embed(query)
        best_score = max(
            self._cosine_similarity(query_embedding, self.embedder.embed(item.chunk.text))
            for item in context.retrieved_chunks[:3]
        )
        low_relevance = best_score < self.threshold

        return GuardrailFinding(
            guardrail_name=self.name,
            triggered=low_relevance,
            severity=Severity.WARNING if low_relevance else Severity.INFO,
            action=Action.WARN if low_relevance else Action.ALLOW,
            message=(
                f"retrieval relevance {best_score:.2f} below threshold {self.threshold:.2f}"
                if low_relevance else
                f"retrieval relevance {best_score:.2f} meets threshold {self.threshold:.2f}"
            ),
            metadata={
                "retrieval_relevance_score": round(best_score, 4),
                "low_retrieval_relevance": low_relevance
            }
        )

    def _cosine_similarity(
        self,
        first: list[float],
        second: list[float]
    ) -> float:
        numerator = sum(a * b for a, b in zip(first, second, strict=True))
        first_norm = sum(a * a for a in first) ** 0.5
        second_norm = sum(b * b for b in second) ** 0.5

        if first_norm == 0 or second_norm == 0:
            return 0.0

        return numerator / (first_norm * second_norm)
