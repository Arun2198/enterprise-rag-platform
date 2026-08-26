# Chapter 4: Retrieval & Reranking

Chapter 3 covered how chunks get turned into vectors and stored. This chapter covers what happens
when a question comes in: how the system decides which stored chunks are actually relevant.

## 1. Why pure vector search isn't enough on its own

Semantic (embedding-based) search is powerful but imperfect. Two common failure modes:

- **Rare or exact terms.** If a user asks about a specific product code, error message, or exact
  legal clause number, an embedding model — trained to capture general meaning — may not weight
  that exact token heavily enough. Plain keyword overlap catches this reliably; pure semantic
  search sometimes doesn't.
- **Semantic drift at the edges.** Two passages can be topically similar (same general subject)
  without being the specific answer to the question asked.

The fix used here is **hybrid retrieval**: combine semantic (vector) search with old-fashioned
keyword overlap, so each covers the other's weak spot.

## 2. `HybridRetriever`

`src/rag/retrieval/hybrid_retrieval.py`. Constructed with two weights (defaults shown):

```python
HybridRetriever(vector_store, embedder, vector_weight=0.65, keyword_weight=0.35)
```

`retrieve(query, top_k, metadata_filter=None)` does the following:

1. Embeds the query (same embedder used at ingestion time — Chapter 3).
2. Asks the vector store for the top `max(top_k * 4, top_k)` candidates by cosine similarity —
   deliberately **over-fetching** more than the final `top_k`, so keyword scoring has a wider pool
   to re-rank within rather than being stuck with whatever the vector search alone considered
   best.
3. For each candidate, computes a **keyword score**: the fraction of query terms present in the
   chunk, log-dampened so that matching, say, 8 out of 10 query words doesn't score dramatically
   higher than matching 4 out of 10 — `math.log1p(overlap) / math.log1p(query_term_count)`.
4. **Fuses** the two scores: `score = 0.65 * vector_score + 0.35 * keyword_score`.
5. Sorts by fused score, returns the top `top_k`.

### Worked example

Query: `"contractor leave days"` → query terms `{contractor, leave, days}`.

Suppose two candidate chunks come back from vector search:

| Chunk | vector_score | text contains | keyword overlap | keyword_score | fused score (0.65v + 0.35k) |
|---|---|---|---|---|---|
| A: "Contractors receive 10 days of leave." | 0.78 | contractor(s), leave, days | 3/3 | `log1p(3)/log1p(3) = 1.0` | `0.65×0.78 + 0.35×1.0 = 0.857` |
| B: "Employees receive 20 days of paid time off annually." | 0.81 | days | 1/3 | `log1p(1)/log1p(3) ≈ 0.50` | `0.65×0.81 + 0.35×0.50 = 0.701` |

Chunk B actually scored *higher* on pure vector similarity (0.81 vs 0.78 — both are about
"time off," so they're semantically close), but Chunk A wins the fused ranking because it's the
one that actually mentions contractors and matches the query's specific terms. This is hybrid
retrieval doing its job: catching a case where semantic similarity alone would have promoted the
topically-similar-but-wrong passage.

## 3. Reranking: a second, more careful pass

Even hybrid retrieval has a structural limitation: both the vector score and the keyword score are
computed **independently** for the query and each chunk (this is called a **bi-encoder**
approach — the query is embedded once, each chunk was embedded once, and they're compared after
the fact). That's fast — it's why the vector store can hold thousands of pre-computed chunk
embeddings and just compare against them — but it can miss things that only become obvious when
the query and the chunk are considered *together*: negation ("chunks NOT mentioning X"),
comparisons ("higher than Y"), word order, or numeric/temporal constraints.

A **cross-encoder** fixes this by taking the query and one candidate chunk *together*, as a single
input, and outputting one relevance score for that pair — much more accurate, but much slower
(it has to run a full forward pass per query-chunk pair, so it can't be pre-computed like an
embedding can). The practical pattern: use the cheap bi-encoder (hybrid retrieval) to narrow
thousands of chunks down to a small candidate set, then use the expensive cross-encoder to
carefully re-score just that small set.

`CrossEncoderReranker` (`src/rag/retrieval/reranker.py`), default model
`cross-encoder/ms-marco-MiniLM-L-6-v2`:

```python
def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    pairs = [(query, candidate.chunk.text) for candidate in candidates]
    scores = self.model.predict(pairs, activation_fn=sigmoid)   # joint (query, chunk) scoring
    return sorted(..., key=score, reverse=True)[:top_k]
```

The raw model output is a logit (an unbounded real number); it's passed through a sigmoid function
to squash it into a bounded `[0, 1]` relevance score that preserves ranking order. That score
**replaces** `RetrievedChunk.score` entirely — the reranker's judgment is what downstream code
(the generation prompt, the `AskResponse.confidence` field) actually sees, not the original fused
hybrid score. It's also stashed separately in `chunk.metadata["cross_encoder_score"]` in case
something needs the raw pre-normalized value.

## 4. How the two stages combine in `RAGService`

`RAGService._retrieve()` (`src/app/services/rag_service.py`) is the orchestration point:

```python
def _retrieve(self, query, top_k, client_id=None):
    if self.reranker is None or not self._reranker_enabled_for(client_id):
        return self.retriever.retrieve(query=query, top_k=top_k)

    candidates = self.retriever.retrieve(query=query, top_k=top_k * self.candidate_multiplier)
    return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
```

So for a request asking for `top_k=5` results, with the default `candidate_multiplier=4`
(`RERANKER_CANDIDATE_MULTIPLIER`): `HybridRetriever` first returns its best 20 candidates (itself
already having over-fetched internally at the vector-store level, per section 2), then the
cross-encoder rescoresall 20 and keeps only the best 5. Two independent over-fetch multipliers are
stacked here — `HybridRetriever`'s internal `top_k * 4` at the vector-store layer, and
`RAGService`'s own `top_k * candidate_multiplier` before reranking — each widening the funnel one
stage earlier so a chunk that a cruder upstream signal underrated still has a chance to be
correctly promoted by a more careful downstream signal.

When `reranker=None` (the default for direct `RAGService()` construction — only
`service_factory.build_rag_service()` turns reranking on) or a feature flag disables it for this
particular caller, `_retrieve()` falls straight back to plain hybrid retrieval with no reranking
step at all — behavior identical to before the reranker existed.

## 5. Feature-flagged rollout

`RAGService` optionally takes a `feature_flags: FeatureFlagManager | None`
([Chapter 8](08-mlops-platform.md) covers this component in full). When set, `_retrieve()` checks
`is_enabled_for("cross_encoder_reranker", client_id)` before using the reranker on each request —
this lets an operator roll reranking out to, say, 25% of traffic and watch metrics before going to
100%, using `client_id` (an optional field on `AskRequest`) as a stable bucketing key so the same
caller consistently lands on the same side of the rollout instead of flickering between requests.
If no flag by that name has been defined yet, the check **fails open** (reranker stays enabled) —
a missing flag definition is treated as a caller/config error, not a silent way to accidentally
disable reranking for everyone. When `feature_flags` is `None` (its default), the reranker simply
runs unconditionally whenever one is configured, with no flag check at all.

Next: [Chapter 5 — Generation: LLMs, Answerers, and Fallback](05-generation-and-llms.md).
