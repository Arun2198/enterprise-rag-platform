# Chapter 7: Evaluation Framework

Everything so far has been about how the pipeline works. This chapter is about how you actually
find out whether it's working *well* — with numbers, not impressions. This is `src/evaluation/`
and the top-level `evaluation/` directory, and it's deliberately standalone from `RAGService`: it
builds its own fresh ingestion/chunking/retrieval pipeline per run, so it can sweep parameters
(chunk size, embedder, reranker on/off) independently of whatever the live app happens to be
configured with.

## 1. Why "it gave an answer" isn't good enough

It's easy to eyeball a RAG system with a handful of questions and conclude it "seems fine." That
doesn't scale, isn't repeatable, and can't catch a regression — exactly what happened with the
chunking bug in [Chapter 2](02-ingestion-and-chunking.md): the system kept returning *an* answer
the whole time, it just quietly got worse at picking the right source for it. You need a
repeatable benchmark: a fixed set of questions with known-correct answers, and a way to score
against them automatically, every time something changes.

## 2. The golden dataset

A **golden dataset** is a set of questions paired with the *known-correct* chunk ids that should
be retrieved for each one. This project's is
[`evaluation/golden_dataset.json`](../../evaluation/golden_dataset.json) — 20 real questions
grounded in `sample_documents/AI-RMF-1stdraft.pdf`. Structure (validated against
`GoldenQuery`/`GoldenDataset` in `src/evaluation/schemas.py`):

```json
{
  "name": "ai-rmf-1stdraft",
  "source_documents": ["sample_documents/AI-RMF-1stdraft.pdf"],
  "queries": [
    {
      "id": "Q001",
      "query": "When was this initial draft of the AI RMF released?",
      "relevant_chunk_ids": ["AI-RMF-1stdraft:1"],
      "category": "framework-overview",
      "difficulty": "easy"
    }
  ]
}
```

**Critical caveat, stated explicitly in the dataset file itself**: `relevant_chunk_ids` like
`"AI-RMF-1stdraft:1"` are **positional** — tied exactly to `chunk_index` from
[Chapter 2](02-ingestion-and-chunking.md)'s `RecursiveChunker`, at one specific set of chunking
parameters. They are not content-addressed (they don't search for "whichever chunk currently
contains this text") — they mean "the chunk that landed at index 1" for that document under those
exact settings. Change the chunker's `chunk_size` or fix a bug in how sections are split, and every
one of these ids can point at a different piece of text than intended, or nothing at all — which
is precisely what happened when [Chapter 2](02-ingestion-and-chunking.md)'s TOC-shredding fix
changed the sample PDF's chunk count from 209 to 148 and every one of these 20 ids had to be
manually re-verified and rewritten against the corrected output.

**How you'd build one for real**: ingest and chunk your source document with the exact chunker
settings you intend to evaluate with, inspect the resulting `(chunk_id, text)` pairs, and hand-pick
which chunk ids actually answer each question you write. There's no shortcut that avoids looking
at real chunk output — fabricating plausible-looking ids without inspecting them produces a
dataset that silently measures nothing.

`evaluation.dataset.load_dataset(path)` validates a dataset file and raises a specific,
locatable `DatasetValidationError` (missing field, empty query list, duplicate id, an invalid
`difficulty` value) rather than letting a malformed file fail deep inside the evaluation run with a
confusing error.

## 3. Layer 1 — retrieval metrics

`src/evaluation/metrics.py`. Given the chunk ids a query's retrieval actually returned
(`retrieved_ids`) and the golden set's correct ids (`relevant_ids`), at some cutoff `k`:

- **`recall_at_k`** — what fraction of the *correct* chunks were found in the top `k` results.
  `len(retrieved_at_k ∩ relevant_ids) / len(relevant_ids)`. Answers "did we find everything we
  should have?"
- **`precision_at_k`** — what fraction of the top `k` results were *actually* correct.
  `hits / k`. Answers "how much of what we returned was noise?"
- **`hit_rate_at_k`** — binary: did *at least one* correct chunk appear in the top `k`? Useful
  because most questions in this dataset have exactly one relevant chunk, where recall and hit
  rate collapse to the same thing.
- **`ndcg_at_k`** (Normalized Discounted Cumulative Gain) — like recall, but rewards finding the
  right chunk *earlier* in the ranking more than finding it near the bottom of the top-k, via
  `1/log2(position+2)` position discounting, normalized against the best-possible ordering (IDCG).
- **`mean_reciprocal_rank`** — averages `1/rank` of the *first* correct hit across all queries in
  the dataset (whole-dataset aggregate, not per-query) — heavily rewards getting the right answer
  into position 1 specifically.
- **`average_rank`** — the mean 1-indexed position of each query's first correct hit, but only
  over queries that had at least one hit — a query with zero hits is *excluded*, not counted as an
  infinite/worst-case rank, so this number stays meaningful; check `hit_rate` alongside it to see
  how many queries that exclusion affects.

### Worked example

Golden answer for a query: `relevant_ids = {"doc:5"}`. Retrieval returns
`["doc:9", "doc:5", "doc:2", "doc:1"]` (in ranked order). At `k=3`:

- `recall_at_3` = `doc:5` is in the top 3 → `1/1 = 1.0`
- `precision_at_3` = 1 correct out of 3 returned → `0.333`
- `hit_rate_at_3` = `1.0` (found it)
- reciprocal rank = it's at position 2 → `1/2 = 0.5`
- `ndcg_at_3` = the hit is discounted for not being position 1: `dcg = 1/log2(2+2) = 0.5`,
  `idcg = 1/log2(1+2) ≈ 0.631` (best case: hit at position 1) → `ndcg ≈ 0.5/0.631 ≈ 0.792`

## 4. Running an evaluation

```bash
uv run python evaluation/run_eval.py --dataset evaluation/golden_dataset.json --k 1 3 5 --json --csv
```

`--provider` picks the embedder (`hashing` or a real sentence-transformers model name),
`--reranker` turns on cross-encoder reranking for the run. Reports land in
`EVALUATION_REPORT_DIR` (default `evaluation/reports/`) as timestamped `.json`/`.csv` files —
these are run artifacts, gitignored except a `.gitkeep`, not source code.

## 5. Comparing configurations: `BenchmarkRunner`

`src/evaluation/benchmark.py`. Runs the *same* golden dataset against several different pipeline
configurations, each with its own freshly built, isolated pipeline (never sharing state across
configs):

```python
BenchmarkRunner(dataset).run([
    BenchmarkConfig(label="baseline", embedder_name="hashing", use_reranker=False),
    BenchmarkConfig(label="bge+reranker", embedder_name="BAAI/bge-base-en-v1.5", use_reranker=True),
])
```

`render_comparison_table(results)` prints them side by side. **The caveat from section 2 applies
directly here**: because `relevant_chunk_ids` are positional, comparing different `chunk_size`
values against one fixed golden dataset makes recall collapse toward zero for every
non-matching size — that's the *correct*, expected outcome of id-based relevance, not a bug in the
benchmark. This dimension is genuinely useful for comparing embedder choice or reranker on/off at
a fixed chunk size; comparing chunk sizes themselves needs a dataset rebuilt (or scored by chunk
*text* rather than id) for each size under test.

## 6. Regression detection

`evaluation.report.compare_reports(current, baseline, threshold=0.02)` diffs two already-written
JSON reports metric by metric — a `MetricDelta.is_regression` fires when a metric dropped by more
than the threshold. `run_eval.py --baseline <path>` runs this automatically and exits with code
`1` if anything regressed, which is meant to be wired into CI as a gate (this repo's CI doesn't
currently do that, but the mechanism is there and tested). This is intentionally just a two-report
diff — not a trend/dashboard system, which is what Layer 4 (below) is for.

## 7. Layer 2 — generation quality

`src/evaluation/generation_metrics.py`, implementing the `GenerationMetric` Protocol
(`score(query, answer, retrieved_chunk_texts) -> float`):

- **`GroundednessMetric`** — the same token-overlap + optional embedding-cosine approach as
  `HallucinationDetector` ([Chapter 6](06-guardrails-and-safety.md)), reimplemented independently
  here (not imported from `rag.guardrails`) so the evaluation package stays fully standalone from
  the app.
- **`AnswerRelevanceMetric`** — cosine similarity between the query's embedding and the answer's
  embedding: does the answer actually address what was asked, regardless of whether it's grounded?
- **`ContextRelevanceMetric`** — token overlap between the query and the retrieved chunks
  themselves (not the answer) — a reference-free check of whether retrieval found on-topic
  material at all.
- **`LLMJudgeGenerationMetric`** (opt-in) — reuses the `OpenAICompatibleAnswerer` client pattern
  to ask an LLM to score generation quality. Fails open by returning `NaN` (not `0.0`) on any
  judge failure, so an outage *excludes* that query from the aggregate mean instead of dragging
  the average down as if it scored zero.

RAGAS-style Context Precision/Recall are deliberately *not* reimplemented here — Layer 1's
`recall_at_k`/`precision_at_k` against hand-verified golden ids is already a more reliable signal
than inferring chunk relevance from the generated answer's text.

`EvaluationRunner` (`src/evaluation/runner.py`) optionally takes an `answer_fn` and a list of
`generation_metrics`; when both are supplied, it runs Layer 2 scoring per query (sliced to
`generation_top_k`, default `min(k_values)`) alongside Layer 1's retrieval metrics, and folds the
result into `aggregate_metrics["generation/{name}"]` (NaN-scored queries excluded from that mean,
the same pattern `average_rank` uses for zero-hit queries).

## 8. Layer 3 — system metrics

`src/evaluation/system_metrics.py::DefaultSystemMetricsCollector.collect()` derives what it can
from a completed `EvaluationReport` alone — query count, retrieval throughput, an *estimated*
completion-token/cost figure from answer text length (explicitly documented as a lower bound,
since the retrieved context text itself isn't retained on `QueryEvaluation`) — plus wall-clock
duration and peak memory when a caller actually measures them (`--system-metrics` wraps the CLI
run in `tracemalloc` + a timer). It deliberately does **not** report a guardrail-trigger rate:
`EvaluationRunner` never executes guardrails at all, so there's no real data behind that number —
reporting an always-zero placeholder would be actively misleading, so it's omitted instead.

## 9. Layer 4 — experiment tracking

`src/evaluation/experiment_tracker.py::LocalExperimentTracker` appends a trimmed
`ExperimentRecord` (config + `aggregate_metrics`, not the full report — full reports are already
saved separately via `--json`/`--csv`) to a local JSON history file
(`evaluation/reports/experiment_history.json`, gitignored) on every tracked run
(`--track --track-path ...`). `--trend N` prints a console table of the last N tracked runs'
metric trend for this dataset (`render_trend_table`). "Trend visualization" here means exactly
that table plus the underlying `MetricTrend` data structure — no charting library or dashboard is
part of this; something else could render that data into a real dashboard later without changing
anything about how it's tracked.

Next: [Chapter 8 — MLOps Platform](08-mlops-platform.md).
