# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sample data

`sample_documents/AI-RMF-1stdraft.pdf` (NIST AI Risk Management Framework, Initial Draft,
March 2022) backs `main.py`'s demo run and `tests/unit/test_pdf_parser.py::test_pdf_parser_success`.

(`src/ingestion/contracts/` and `src/ingestion/parsers/base_parser.py` + `pdf_parser.py` were
previously missing from git — only stale `__pycache__/*.pyc` files had been committed — and were
reconstructed by disassembling that bytecode; see commit history around this fix.)

## Commands

Dependencies are managed with `uv` (see `uv.lock`); there's no separate lint/format tool configured.

```bash
# install/sync dependencies
uv sync

# run the full test suite
uv run pytest

# run a single test file / test
uv run pytest tests/unit/test_rag_service.py
uv run pytest tests/unit/test_rag_service.py::test_name -v

# run the demo script (ingests a sample PDF and asks a question end-to-end)
uv run python main.py

# run the FastAPI service locally
uv run uvicorn app.main:app --reload --app-dir src

# run retrieval evaluation against the golden dataset
uv run python evaluation/run_eval.py --dataset evaluation/golden_dataset.json --k 1 3 5 --json
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so tests import top-level packages
directly (`from app...`, `from ingestion...`, `from rag...`, `from evaluation...`) rather than
`from src...`.

## Architecture

This is an MVP Retrieval-Augmented Generation service with a strict layered pipeline and a
provider-swap pattern used throughout: every stage (parsing, embedding, vector storage,
generation) is defined as a `Protocol` with a local, dependency-free default implementation, plus
commented-out/injectable production adapters. Nothing is wired to a DI container — swapping
providers means passing a different instance into `RAGService.__init__`.

**Request flow:** `app/main.py` (FastAPI routes `/ingest`, `/ask`, `/health`) → `RAGService`
(`app/services/rag_service.py`), which owns the whole pipeline:

1. **Ingestion** (`src/ingestion/`) — `IngestionPipeline.ingest_file()` picks a parser via
   `ParserFactory` (dispatches on file extension: `.pdf`/`.docx`/`.md`/`.markdown`), parses to a
   `Document`, then normalizes whitespace via `TextCleaner`. Every parser and pipeline stage
   returns a `Result[T]` (success/data/error) instead of raising — errors carry a `code` +
   `message` and are surfaced up through `RAGService.ingest()` as strings in
   `IngestResponse.errors`, never as exceptions.
2. **Chunking** (`src/rag/chunking/recursive_chunker.py`) — `RecursiveChunker` splits a
   `Document` into `Chunk`s: first by heading-like lines (`_looks_like_heading`), then within each
   section by sentence boundaries up to `chunk_size` with `chunk_overlap`, merging any resulting
   chunk under `minimum_chunk_size` into its neighbor. Chunk metadata carries the parent section
   title and document metadata forward.
3. **Embedding** (`src/rag/embeddings/`) — `Embedder` protocol. Default is `HashingEmbedder`, a
   deterministic hash-based bag-of-tokens embedding (no external model/credentials needed, keeps
   the MVP runnable offline). `SentenceTransformerEmbedder` is a real local-model alternative
   (BAAI/bge-small-en-v1.5), not wired into `service_factory`.
4. **Vector store** (`src/rag/vector_store/`) — `VectorStore` protocol. `InMemoryVectorStore` does
   brute-force cosine similarity over an in-process dict, used for local/dev/tests.
   `OpenSearchVectorStore` is the production adapter; it takes an already-authenticated OpenSearch
   client injected by the caller (never constructs its own client, to keep the core package free
   of AWS/OpenSearch imports).
5. **Retrieval** (`src/rag/retrieval/hybrid_retrieval.py`) — `HybridRetriever` fuses vector cosine
   similarity with a keyword-overlap score (`vector_weight=0.65` / `keyword_weight=0.35`) over an
   over-fetched candidate set (`top_k * 4`), then truncates to `top_k`.
6. **Reranking** (`src/rag/retrieval/reranker.py`) — `CrossEncoderReranker`, on by default
   (`RERANKER_ENABLED=true`). `RAGService._retrieve()` over-fetches
   `top_k * RERANKER_CANDIDATE_MULTIPLIER` (default multiplier 4) from `HybridRetriever`, then the
   reranker jointly scores each `(query, chunk_text)` pair with a cross-encoder
   (`cross-encoder/ms-marco-MiniLM-L-6-v2` by default) and keeps only the best `top_k` — this
   catches negation/comparisons/word-order that independent bi-encoder retrieval scoring misses.
   The cross-encoder score becomes the `RetrievedChunk.score` used downstream for confidence and
   is also stashed in `chunk.metadata["cross_encoder_score"]`. When `RAGService` is constructed
   with `reranker=None` (its own default — only `service_factory` turns reranking on), `_retrieve`
   falls straight back to a plain `HybridRetriever.retrieve(top_k)` call, unchanged from before
   this stage existed. `tests/unit/conftest.py` patches the underlying `sentence_transformers
   .CrossEncoder` for the whole test session so no unit test ever downloads the real model.
7. **Generation** (`src/rag/generation/`) — `Answerer` protocol. `ExtractiveAnswerer` picks the
   best-overlap sentence from retrieved chunks (no LLM call, fully deterministic — grounded by
   construction since it only ever returns retrieved text). `BedrockAnswerer` and
   `OpenAICompatibleAnswerer` are LLM-backed adapters that share one grounded prompt template
   (`rag/generation/prompt.py::build_grounded_prompt`) so every provider answers only from
   retrieved context and cites the same source chunk ids. `BedrockAnswerer` takes an injected
   boto3 `bedrock-runtime` client. `OpenAICompatibleAnswerer` builds its own `openai.OpenAI`
   client from `api_key`/`base_url`/`model_name` and works against any OpenAI-compatible Chat
   Completions endpoint (OpenAI, Azure OpenAI, GitHub Models, Ollama, OpenRouter, Groq, ...) by
   changing only those config values — no code changes. It returns a fixed fallback string
   without calling the LLM when there are no retrieved chunks, retries only transient failures
   (HTTP 429/500/502/503/504) with exponential backoff, and returns a fallback string on
   exhausted/non-retryable failures instead of raising.
8. **Guardrails** (`src/rag/guardrails/`) — every `Guardrail` (`base.py`) implements one
   `check(context) -> GuardrailFinding` method and declares a `stage` (`INPUT` or `OUTPUT`).
   `GuardrailManager` (`manager.py`) runs the guardrails registered for a given stage, applies any
   redactions in sequence, and resolves the strictest `Action` (`ALLOW < WARN < REDACT < ESCALATE
   < BLOCK`) across triggered findings. `RAGService.ask()` calls `run_input(query)` before
   retrieval (a `BLOCK` there short-circuits before any retrieval/generation happens) and
   `run_output(query, answer, retrieved_chunks)` after generation (a `BLOCK` there replaces the
   answer and empties `sources`, so a blocked response never leaks retrieved chunk text). Phase 1
   default guardrails (both output-stage, wired by `GuardrailManager.default()`):
   `PIIGuard` (`pii_guard.py`, regex redaction for email/phone/SSN/credit-card/Aadhaar, PII_GUARD_ENABLED)
   and `HallucinationDetector` (`hallucination_detector.py`, token-overlap groundedness blended
   with embedding cosine similarity when an `Embedder` is available — reuses `RAGService`'s own
   `HashingEmbedder`, no new dependency — HALLUCINATION_GUARD_ENABLED /
   GROUNDEDNESS_THRESHOLD). `AskResponse.guardrail_flags` is built by
   `GuardrailManager._build_flags()`: `pii_detected`/`hallucination`/`groundedness` are flattened
   to match the HLD's example shape, plus a generic `details` list so any other guardrail's
   findings show up automatically without a schema change. Three more lightweight, dependency-free
   guardrails exist and implement the same interface but are **not** in the Phase 1 default set -
   register them explicitly to opt in: `PromptInjectionGuard` (input stage, regex heuristics for
   injection/jailbreak phrasing), `SecretLeakageGuard` (output stage, API key/token/private-key
   patterns), `ProfanityGuard` (output stage, small illustrative wordlist). `PolicyEngine`
   (`policy.py`) evaluates configurable `PolicyRule`s (condition: guardrail name + min severity
   and/or a metadata threshold) that can escalate — never downgrade — the action a `GuardrailManager`
   would otherwise take; `PolicyEngine.default_policies()` implements the HLD's two example
   policies. It is **not** attached by `GuardrailManager.default()` — Phase 1 findings apply their
   own suggested action directly (PII → redact, hallucination → warn, never auto-block); pass
   `policy_engine=` explicitly to opt in.

   Three ML-backed guardrails also exist, none in the Phase 1 default set - `service_factory`
   wires each in only when its own `*_ENABLED` flag is set:
   - `PresidioPIIGuard` (`presidio_pii_guard.py`, `PRESIDIO_PII_GUARD_ENABLED`) - Microsoft
     Presidio NER + pattern recognizers instead of plain regex, so it catches names/addresses/
     other context-dependent PII the regex `PIIGuard` structurally can't. Loads a spaCy
     `en_core_web_sm` model once at construction (pinned as a direct `uv` dependency via wheel
     URL in `pyproject.toml`, so `uv sync` alone is enough - no separate `spacy download` step).
     Registered *alongside* `PIIGuard`, not replacing it. Adds a custom Aadhaar
     `PatternRecognizer` since Presidio has none built in. Overlapping spans (Presidio can flag a
     `URL` entity fully inside an `EMAIL_ADDRESS` for the same text) are resolved by preferring
     the higher-confidence, longer span; redaction then applies start-descending so earlier
     replacements never shift not-yet-processed offsets.
   - `NLIHallucinationDetector` (`nli_hallucination_detector.py`, `NLI_HALLUCINATION_ENABLED`,
     default model `cross-encoder/nli-deberta-v3-base`) - same `sentence_transformers
     .CrossEncoder` pattern as the reranker, but scores (chunk, answer) as (premise, hypothesis)
     NLI pairs via `predict(pairs, apply_softmax=True)` and takes the max entailment probability
     across chunks as groundedness (the answer only needs to be entailed by at least one chunk).
     The 3-class label order (0=contradiction, 1=entailment, 2=neutral) is model-specific -
     verified against this model's config.json; check again if the model name is ever changed.
   - `LLMJudgeHallucinationDetector` (`llm_judge_hallucination_detector.py`, `LLM_JUDGE_ENABLED`)
     - reuses `OpenAICompatibleAnswerer`'s client/retry pattern (own `openai.OpenAI` client,
     retries only 429/500/502/503/504) to ask the configured LLM to score groundedness via a
     JSON-only prompt; tolerates markdown-fenced JSON. Fails open (does not trigger, does not
     block) on any API or parse failure, with `metadata["judge_available"] = False` marking that
     case. If `LLM_JUDGE_BASE_URL`/`LLM_JUDGE_API_KEY` aren't set, falls back to the main
     `LLM_BASE_URL`/`LLM_API_KEY`; `service_factory` raises `ServiceConfigurationError` at
     startup if neither pair resolves.

   Toxicity/hate-speech classification, BERTScore, and RAGAS are still not implemented - a
   low-quality toxicity classifier can cause real harm and deserves a deliberate follow-up rather
   than a quick add; BERTScore/RAGAS were judged to mostly duplicate what NLI/LLM-judge already
   cover here. All three would implement the same `Guardrail` interface if added later.

   **Observability** (`telemetry.py`) - `GuardrailManager._run()` calls `record_finding()` /
   `record_action()` for every check via the OpenTelemetry API (`opentelemetry-api` +
   `opentelemetry-sdk` are real dependencies, but no exporter is configured here). Instruments:
   `guardrail.runs` / `guardrail.latency` (every check), `guardrail.pii_detections` (`PIIGuard`
   and `PresidioPIIGuard` triggers), `guardrail.hallucination_flags` +
   `guardrail.groundedness_score` (all three hallucination detectors), `guardrail.blocked_responses`
   (any `Action.BLOCK`). With no `MeterProvider` configured these are cheap no-ops - a host app
   can call `opentelemetry.metrics.set_meter_provider(...)` with a Prometheus or console exporter
   at startup and every metric here starts flowing, retroactively, with zero changes on this side
   (verified in `tests/unit/test_guardrails_telemetry.py`). Recording never raises - a broken
   exporter must not break the guardrail pipeline it's observing. No live Prometheus/Grafana
   server is set up or required by this repo.

**Feature-flagged reranking:** `RAGService` optionally takes a `feature_flags:
mlops.feature_flags.FeatureFlagManager | None`. When set, `_retrieve()` checks
`is_enabled_for("cross_encoder_reranker", client_id)` per request before using the reranker (a
missing flag definition fails open to "enabled" rather than silently disabling reranking for
everyone); when `feature_flags` is `None` (the default when constructing `RAGService` directly),
the reranker runs unconditionally whenever configured, exactly as before this existed. `client_id`
is an optional field on `AskRequest`/param on `RAGService.ask()` used as the canary bucketing
subject for stable per-caller rollout; a random id is used per-request when omitted (fine for an
anonymous canary sample, just not sticky across requests from the same untracked caller).

**Wiring:** `app/service_factory.py::build_rag_service()` reads `app/config.py::Settings` (from
env vars: `VECTOR_STORE_PROVIDER`, `EMBEDDING_PROVIDER`, `GENERATION_PROVIDER`, `LLM_*`,
`RERANKER_*`, etc.). `GENERATION_PROVIDER` accepts `extractive` (default) or `openai_compatible`
(requires `LLM_BASE_URL` + `LLM_API_KEY`, raises `ServiceConfigurationError` if either is
missing); any other value raises `ServiceConfigurationError`. `VECTOR_STORE_PROVIDER` and
`EMBEDDING_PROVIDER` only permit their local/default values (`memory`, `hashing`) — the
production adapters (`OpenSearchVectorStore`, `BedrockAnswerer`) must be constructed and injected
by the caller directly, since those need externally-managed authenticated clients the factory
doesn't build. `RERANKER_ENABLED` (default `true`), `RERANKER_MODEL_NAME`, and
`RERANKER_CANDIDATE_MULTIPLIER` control the reranking stage independently of which generation
provider is active — set `RERANKER_ENABLED=false` to bypass it entirely and get the pre-reranking
`HybridRetriever` behavior back. `GUARDRAILS_ENABLED` (default `true`) is the master switch for
the whole guardrails stage — `false` gives `RAGService` an empty `GuardrailManager` (no findings,
`guardrail_flags` comes back as `{}`) regardless of any other guardrail flag; `PII_GUARD_ENABLED`,
`HALLUCINATION_GUARD_ENABLED`, and `GROUNDEDNESS_THRESHOLD` control the Phase 1 defaults
individually. `service_factory._build_guardrail_manager()` (not `RAGService`'s own inline
defaults) is what actually assembles the full list for real app wiring, including the opt-in
ML-backed guardrails: `PRESIDIO_PII_GUARD_ENABLED` / `PRESIDIO_SCORE_THRESHOLD` /
`PRESIDIO_ENTITIES` (comma-separated), `NLI_HALLUCINATION_ENABLED` / `NLI_MODEL_NAME` /
`NLI_THRESHOLD`, and `LLM_JUDGE_ENABLED` / `LLM_JUDGE_BASE_URL` / `LLM_JUDGE_API_KEY` /
`LLM_JUDGE_MODEL_NAME` / `LLM_JUDGE_THRESHOLD` (base_url/api_key fall back to `LLM_BASE_URL`/
`LLM_API_KEY` if unset). `FEATURE_FLAGS_ENABLED` (default `true`) makes `build_rag_service()`
attach a `FeatureFlagManager` with the `cross_encoder_reranker` flag pre-defined at
`RERANKER_ROLLOUT_PERCENTAGE` (default `100`, i.e. unchanged behavior); `false` leaves
`RAGService.feature_flags` as `None`.

`app/main.py::build_platform_manager(settings)` builds the shared `mlops.manager.PlatformManager`
for the live app (returns `None` when `MLOPS_ENABLED=false`) and is passed into
`build_rag_service(settings, platform_manager=...)` so the app's `FeatureFlagManager` and the one
admin endpoints operate on are the same instance - updating a flag via the API actually changes
`/ask` behavior, not just a disconnected copy. When `SCHEDULER_ENABLED`, `main.py` registers a
`backup` job (snapshots registry/artifacts/configuration/feature_flags to
`SCHEDULER_BACKUP_DIR`, default `mlops_backups/`) and a `health_check` job (logs indexed chunk
count) on `platform_manager.scheduler`, then drives `run_due_jobs()` from an `asyncio` task
started in the FastAPI `lifespan` context manager every `SCHEDULER_INTERVAL_SECONDS` (default
`300`) - this is the "whatever actually owns scheduling in a deployment" piece `Scheduler` itself
deliberately doesn't provide. Admin endpoints: `GET /admin/feature-flags`, `PATCH
/admin/feature-flags/{name}` (body: `enabled`?/`rollout_percentage`?), `GET
/admin/scheduler/jobs`, `POST /admin/scheduler/jobs/{job_id}/trigger` (runs a job immediately via
`Scheduler.trigger()`); all four 404 when `platform_manager` is `None`.

Tests in `tests/unit/` mirror this structure one-to-one and mostly test each layer in isolation
via real (non-mocked) local implementations, plus `test_api.py` for an end-to-end
ingest-then-ask flow through the FastAPI `TestClient`.

## Evaluation framework

Standalone from the app (`src/evaluation/`, top-level `evaluation/`) — a golden-dataset-driven
retrieval benchmark, not wired into `RAGService`. It builds its own fresh ingestion/chunking/
retrieval pipeline per run rather than reusing `RAGService`, since it needs to sweep
chunk_size/embedder/reranker independently of whatever `GENERATION_PROVIDER`/`RERANKER_ENABLED`
the app happens to be configured with.

**Pipeline:** Golden Dataset → `EvaluationRunner` → `retrieve_fn(query, top_k) -> list[chunk_id]`
→ `metrics.py` → `EvaluationReport` → console/JSON/CSV. `EvaluationRunner` (`runner.py`) is
deliberately decoupled from `HybridRetriever`/`RAGService` — it only needs a plain callable, so
the same runner works against a raw retriever, a reranked pipeline, or a test double. Adding a
metric means adding a function to `metrics.py` and one line each in `EvaluationRunner
._compute_query_metrics`/`_aggregate`; nothing about the runner's shape changes.

**How to create a dataset** — JSON matching `schemas.GoldenQuery`/`GoldenDataset`
(`name`, `source_documents`, `queries[]` with `id`/`query`/`relevant_chunk_ids`/`category`?/
`difficulty`? — difficulty must be `easy`/`medium`/`hard` if present).
`dataset.load_dataset(path)` validates and raises `DatasetValidationError` with a specific,
locatable message (missing field, empty list, duplicate id, bad difficulty, ...) rather than
letting a malformed dataset fail deep inside the runner. `relevant_chunk_ids` are exact
`"{document_id}:{index}"` strings from `RecursiveChunker` — **positional, not content-addressed**,
so they're only valid for the exact chunking parameters used to build them. To build a real
dataset: ingest+chunk the source document with the chunker settings you intend to evaluate with,
inspect the resulting `chunk_id`/`text` pairs, then hand-pick relevant ids per query (this is
literally how `evaluation/golden_dataset.json` — 20 queries grounded in
`sample_documents/AI-RMF-1stdraft.pdf` at `RecursiveChunker()` defaults — was built; don't
fabricate ids without inspecting real chunks). `tests/unit/test_evaluation_integration.py` runs
this dataset through the real PDF end-to-end and asserts `recall@10 > 0.5` specifically so a
future chunker default change that invalidates the ids fails loudly instead of silently.

**How to add a metric** — add a `(retrieved_ids: list[str], relevant_ids: set[str], k: int) ->
float` function to `metrics.py` (per-query) or match `mean_reciprocal_rank`'s shape
(`list[list[str]], list[set[str]] -> float`, whole-dataset aggregate) for something that isn't
naturally per-query. Existing: `recall_at_k`, `precision_at_k`, `hit_rate_at_k`, `ndcg_at_k`
(binary relevance, standard DCG/IDCG), `mean_reciprocal_rank`, `average_rank` (mean 1-indexed
position of each query's first hit; queries with zero hits are excluded, not penalized as
infinite), `average_retrieved_documents`.

**How to run evaluation** — `uv run python evaluation/run_eval.py --dataset
evaluation/golden_dataset.json --k 1 3 5 --json --csv` (`--provider` picks the embedder:
`hashing` or any sentence-transformers model name; `--reranker` enables the cross-encoder).
Reports land in `EVALUATION_REPORT_DIR` (default `evaluation/reports/`, gitignored except
`.gitkeep` — these are run artifacts, not source) as
`{dataset_name}_{YYYYMMDD_HHMMSS}.{json,csv}`.

**How to benchmark retrieval configurations** — `benchmark.BenchmarkRunner(dataset).run([
BenchmarkConfig(label=..., chunk_size=..., chunk_overlap=..., minimum_chunk_size=...,
embedder_name=..., use_reranker=...), ...])` builds one fresh, isolated pipeline per config
(never cross-contaminated) and returns `list[(config, EvaluationReport)]`;
`render_comparison_table(results)` prints a side-by-side table. **Caveat that matters**: since
`relevant_chunk_ids` are positional, comparing `chunk_size`/`chunk_overlap` across configs against
one fixed golden dataset will show recall collapsing toward zero for every non-matching size —
that's correct behavior given ID-based relevance, not a bug (see `BenchmarkConfig`'s docstring).
This dimension is meaningful for comparing embedder/reranker/hybrid choices at a *fixed* chunk
size; comparing chunk sizes themselves needs a dataset rebuilt (or judged by chunk text, not id)
per size under test.

**How to compare experiments (regression detection)** — `report.compare_reports(current,
baseline, threshold=0.02)` diffs two already-written JSON reports metric-by-metric
(`MetricDelta.is_regression` when `delta < -threshold`); `--baseline <path>` on the CLI runs this
automatically and exits `1` if any metric regressed (wire that exit code into CI if you want a
gate). This is intentionally just a two-report diff, not a trend/dashboard system — see
`ExperimentTracker` below.

**Layer 2 (generation quality)** — `generation_metrics.py` implements `GenerationMetric`:
`GroundednessMetric` (token-overlap + optional embedding cosine, same scoring approach as
`rag.guardrails.hallucination_detector.HallucinationDetector`, kept as an independent
implementation rather than an import so evaluation stays standalone), `AnswerRelevanceMetric`
(query/answer embedding cosine), `ContextRelevanceMetric` (query/chunk token overlap, reference-
free), and `LLMJudgeGenerationMetric` (opt-in, reuses the `OpenAICompatibleAnswerer` client
pattern; fails open by returning `NaN`, not `0.0`, so a judge outage drops that query from the
aggregate mean instead of tanking it). RAGAS-style Context Precision/Recall are intentionally
*not* duplicated here since Layer 1 already computes those against golden-dataset
`relevant_chunk_ids` (`recall_at_k`/`precision_at_k`), a more reliable signal than judging chunk
relevance from answer text alone. `EvaluationRunner` takes optional `answer_fn:
(query, retrieved_chunk_ids) -> (answer, retrieved_chunk_texts)` and `generation_metrics:
list[GenerationMetric]`; when both are set it also runs Layer 2 per query (sliced to
`generation_top_k`, default `min(k_values)`) and folds results into `QueryEvaluation.answer`/
`.generation_metrics` and `aggregate_metrics["generation/{name}"]` (NaN scores excluded from the
mean, matching how `average_rank` excludes zero-hit queries rather than penalizing them).
`BenchmarkConfig.generation_provider` (`None`/`"extractive"`/`"openai_compatible"`) wires this
into `BenchmarkRunner` automatically; CLI flag `--generation extractive|openai_compatible`.

**Layer 3 (system metrics)** — `system_metrics.py::DefaultSystemMetricsCollector.collect(report,
run_duration_seconds=None, peak_memory_mb=None)` computes what's derivable from an
`EvaluationReport` alone (query count, retrieval throughput, estimated completion tokens/cost from
answer text length - documented as a lower bound since retrieved context text isn't retained on
`QueryEvaluation`), plus `run_duration_seconds`/`peak_memory_mb` only when a caller supplies real
measurements (CLI flag `--system-metrics` wraps the run in `tracemalloc` + a wall clock). Does
**not** report a guardrail-trigger rate - `EvaluationRunner` never executes guardrails, so there's
no real data to summarize; fabricating an always-zero metric would be worse than omitting it.

**Layer 4 (experiment tracking)** — `experiment_tracker.py::LocalExperimentTracker` is an
append-only JSON history file (`evaluation/reports/experiment_history.json` by default,
gitignored) of trimmed `ExperimentRecord`s (metadata + `aggregate_metrics`, not full reports -
those are already saved separately by `--json`/`--csv`). `record()`/`history()`/
`trend_from_history()` read/write the file; `compare_many(reports)` (the `ExperimentTracker`
Protocol method) builds the same `MetricTrend` shape directly from in-hand reports without
touching the file. "Trend visualization" means a console table
(`experiment_tracker.render_trend_table`) plus the raw `MetricTrend`/`MetricTrendPoint` data - no
charting/dashboard dependency is added; something else can render that data if a real dashboard is
ever wanted. CLI flags `--track` (record this run) / `--track-path` / `--trend N` (print a trend
table over the last N tracked runs for this dataset).

## Operations & MLOps platform (`src/mlops/`)

A provider-agnostic operational backbone (asset registries, lifecycle, config, feature flags,
secrets, scheduling, governance, backup/recovery, RBAC). Every stateful component is in-memory by
default and has no persistence unless explicitly backed up (`BackupManager`) - this is a library,
not a service with its own database. Registries/artifacts/lifecycle/governance stay standalone
from the app (nothing in the request path writes to them yet - see per-section notes below for
what a real caller would do with each). Feature flags and the scheduler *are* wired into the live
app now - see "Wiring" and "Feature-flagged reranking" in the main Architecture section above for
how `app/main.py`/`service_factory.py` share one `PlatformManager` instance with `RAGService` and
the admin endpoints.

**Architecture** — `manager.PlatformManager` is a thin facade composing the components below;
each is equally usable standalone (`from mlops.registry import ModelRegistry` works with zero
platform-manager ceremony). `PlatformManager` exists for cross-component workflows - `promote()`
does a `LifecycleManager` transition *and* mirrors it into `GovernanceLog` in one call - and for
`register_provider(name, obj)`/`get_provider(name)`, a single named slot any pluggable backend
(a real MLflow registry, a cloud secrets client, a CI pipeline, a drift detector) registers into.

**Registries** — `registry.ModelRegistry` tracks versioned AI assets (embedding models,
rerankers, LLM providers, prompt templates, guardrail models, evaluation models) keyed
`"{asset_type}:{name}:{version}"`, each carrying a `LifecycleStage` status and free-form
metadata. `artifacts.ArtifactRegistry` is separate and **immutable, append-only** - prompt
templates, chunking/embedding configs, eval datasets, experiment definitions, policies,
guardrail configs, feature definitions - `save()` never overwrites, it always creates version
`N+1`; nothing already saved is ever mutated. `ModelRegistryBackend` (in `registry.py`) is the
unimplemented extension point a real MLflow/Azure ML/SageMaker/Vertex AI/Kubeflow registry would
satisfy.

**Lifecycle** — `lifecycle.LifecycleManager` is the promotion state machine: `Development ->
Validation -> Staging -> Production -> Retired`, plus the reject-back edges (`Validation ->
Development`, `Staging -> Validation`) - the full legal-transition table is
`ALLOWED_TRANSITIONS`. Promoting into `Staging` or `Production` requires `approved_by`
(raises `ApprovalRequiredError` otherwise); every transition and approval is recorded and
queryable via `.history(asset_id)`/`.approvals(asset_id)`.

**Configuration** — `configuration.ConfigurationManager` holds named environment profiles
(dev/staging/prod/...), each independently versioned and append-only like `ArtifactRegistry`.
`activate(name, version=None)` picks a profile version to read from; `get(key)` checks runtime
overrides first, then the active profile's values, then a caller-supplied default; `rollback()`
re-activates the previous version of whichever profile is (or was) active. Optional per-key
`validators` reject an entire `save_profile()` call if any value fails its validator - nothing
partially-invalid ever enters history.

**Feature flags** — `feature_flags.FeatureFlagManager`: boolean enable/disable, percentage-based
canary rollout, and a `shadow` marker (the flag just tracks shadow state - what "shadow mode"
does is entirely up to the caller's own code). `is_enabled_for(name, subject_id)` uses stable
SHA-256 hash bucketing so the same subject always gets the same answer for a given
flag+percentage, instead of flapping between requests.

**Secrets** — `secrets.SecretsProvider` Protocol + `LocalEnvSecretsProvider` (reads
`os.environ`, optionally namespaced by a `prefix`) as the only implemented backend; Azure Key
Vault/AWS Secrets Manager/GCP Secret Manager are extension points, not implemented (no cloud SDKs
are project dependencies). `secrets.SecretValue` wraps every returned secret so `repr()`/`str()`
always print `***redacted***` - call `.reveal()` explicitly to get the raw string, which should
be the only place a secret value ever touches application code.

**CI/CD** — `deployment.DeploymentPipeline` Protocol (`run_tests`/`run_evaluation`/
`run_experiment`/`deploy`, each returning a `StageResult`). `LocalDeploymentPipeline` is a real,
working reference implementation - `run_tests()`/`run_evaluation()` genuinely shell out to
`pytest`/`evaluation/run_eval.py` via `subprocess`, proving the Protocol is actually usable, not
just a paper interface; `deploy()` only logs intent (actually deploying is inherently
provider-specific). `GitHubActionsDeploymentPipeline` implements the same Protocol against real
GitHub Actions, driven entirely through the `gh` CLI (`workflow run` to dispatch, `run list`/`run
view --json` to poll for the new run and its conclusion) rather than a raw REST client with its
own token handling - reuses whatever `gh auth login` session or `GH_TOKEN`/`GITHUB_TOKEN` is
already in the environment. Each stage maps to an independently configurable workflow file
(`test_workflow`/`evaluation_workflow`/`experiment_workflow`/`deploy_workflow`); a stage left
`None` is a soft no-op (same spirit as `LocalDeploymentPipeline.deploy()`'s placeholder) rather
than forcing every repo to have all four workflows. Not verified against a real `gh` invocation in
this repo (the sandbox this was built in doesn't have `gh` installed) - covered by mocked
`subprocess.run` unit tests only; verify for real in an environment with `gh auth login` done
before relying on it. Azure DevOps/Jenkins/GitLab CI adapters are still **not implemented** - a
real one would call that provider's REST API/CLI per stage but implement this exact same
Protocol.

**Scheduler** — `scheduler.Scheduler` is a real job registry with interval-based due-job
execution, deliberately with **no background thread of its own** - call `run_due_jobs(now)`
periodically from whatever actually owns scheduling in a deployment (a loop, a Kubernetes
CronJob, a GitHub Actions schedule trigger, cron itself). This keeps it dependency-free and
testable with a fake clock instead of needing real sleeping/threading in tests. `trigger(job_id)`
runs a job immediately, outside its schedule. Example jobs: re-index documents, run evaluation,
health checks, drift detection, backup - register any zero-argument callable. The FastAPI app is
one such "whatever actually owns scheduling" - see "Wiring" in the main Architecture section for
the `asyncio`-task-in-`lifespan` loop and the two jobs it registers by default.

**Drift & retraining (not implemented)** — `drift.DriftDetector` Protocol covers `DRIFT_TYPES`
(data, embedding, retrieval, prompt, model, user_query); `retraining.RetrainingTrigger` +
`ValidationWorkflow` Protocols cover `RETRAINING_TRIGGERS` (scheduled, drift, manual). No model
training happens in this repo. A concrete `ValidationWorkflow` would naturally run an
`evaluation.runner.EvaluationRunner` pass and gate on `evaluation.report.compare_reports`, then
hand off to `LifecycleManager` for promotion.

**Governance** — `governance.GovernanceLog` is the audit trail: every `record()`/
`record_transition()`/`record_approval()`/`link_lineage()`/`check_policy()` call appends an
`AuditEvent`, queryable via `.history(resource=...)`. `link_lineage(asset_id, artifact_id,
version)` tracks which artifact versions produced a given model asset, so "what actually went
into this production model" stays answerable. `check_policy(rule_name, condition, message)`
records the outcome either way (a passed check is just as visible as a failed one) and raises
`PolicyViolationError` on failure.

**Backup & recovery** — `backup.BackupManager.create_snapshot({name: component, ...})` writes a
timestamped local JSON file; any component with `.export_state()`/`.import_state()` qualifies
(`ModelRegistry`, `ArtifactRegistry`, `ConfigurationManager`, `FeatureFlagManager` all implement
it). `recovery.RecoveryManager.restore_snapshot(path, components)` restores only the components
explicitly passed in, silently skipping anything else present in the snapshot. `backup.BackupTarget`
(a cloud destination - S3/Azure Blob/GCS) is an extension point, not implemented; `BackupManager`
only ever writes locally.

**Permissions (RBAC, no auth)** — `permissions.py`: `Role` (Administrator/MLEngineer/
DataScientist/Reviewer/ReadOnly) × `Permission` via a static `ROLE_PERMISSIONS` matrix;
`has_permission(role, permission)`/`require_permission(role, permission)` (raises
`PermissionDeniedError`). This only answers "given a role, is X allowed" - establishing *who* the
actor is (login, sessions, tokens) is entirely out of scope and the caller's responsibility.

**Observability** — `mlops/telemetry.py` follows the exact same pattern as
`rag/guardrails/telemetry.py`: OTel API counters (`mlops.operations`, `mlops.audit_events`),
no-op with no `MeterProvider` configured, never raises. `PlatformManager` calls it on every
operation (register/promote/backup/restore) alongside the matching `GovernanceLog` entry, so
metrics and audit trail always move together.

## Architecture rules (do not violate)
- Factory pattern in service_factory.py gates unwired providers via
  ServiceConfigurationError. Never silently wire a new provider without
  updating both the factory AND its guard.
- All cross-service data must match the contracts in schemas.py /
  ingestion/contracts/. Don't invent new fields without updating the
  contract + tests.
- Every new module needs a corresponding test in tests/unit, matching
  the existing style (see test_recursive_chunker.py as the pattern).

## Commands
- Run tests: uv run pytest tests/unit -v
- Run app: uv run python main.py
- Run API: uv run uvicorn app.main:app --reload --app-dir src
- Run evaluation: uv run python evaluation/run_eval.py --dataset evaluation/golden_dataset.json

## Writing style for all code, comments, docstrings, and docs
- Write comments and docstrings the way a working engineer actually writes
  them: brief, practical, occasionally informal. Not exhaustive, not
  textbook-style.
- No mention of AI assistance, Claude, or any AI tool anywhere — not in
  code comments, docstrings, README, commit messages, or PR descriptions.
- No em-dashes or corporate-sounding phrasing ("leverages", "robust
  solution", "seamlessly"). Write like a person explaining a decision to
  a teammate.
- Commit messages: short, conventional (feat/fix/refactor/test prefix),
  no trailers beyond what's normal for a solo dev commit.