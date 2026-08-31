# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Sample data

`sample_documents/AI-RMF-1stdraft.pdf` (NIST AI Risk Management Framework, Initial Draft,
March 2022) backs `main.py`'s demo run and `tests/unit/test_pdf_parser.py::test_pdf_parser_success`.

(`src/ingestion/contracts/` and `src/ingestion/parsers/base_parser.py` + `pdf_parser.py` were
previously missing from git — only stale `__pycache__/*.pyc` files had been committed — and were
reconstructed by disassembling that bytecode; see commit history around this fix.)

## Tech stack

**Custom-built** (this codebase's own code, not a wrapper around someone else's framework):
`RecursiveChunker`, `HybridRetriever` (RRF fusion), the Okapi BM25 implementation, `HashingEmbedder`,
`ExtractiveAnswerer`, `FallbackAnswerer`, `InMemoryVectorStore`, `OpenSearchVectorStore` /
`OpenSearchIndexManager` (the integration layer, not OpenSearch itself), `SignedOpenSearchClient`
(hand-rolled SigV4 HTTP client, written after finding a real hang bug in `opensearch-py`), every
guardrail and `GuardrailManager` itself (including how third-party detectors like Presidio/NLI/
LLM-judge get wired into the framework — the detectors' models are third-party, the integration is
not), `OIDCTokenValidator` and the RBAC `Role`/`Permission` matrix, `InMemoryRateLimiter` (a small
dependency-free fixed-window API rate limiter), all FastAPI routes/schemas, the async S3/SQS
ingestion worker, the `SQSSchedulerWorker` that fixes cross-task scheduled-job double-execution,
the entire `src/mlops/` platform (registries, lifecycle manager, config manager, feature flags,
secrets provider, scheduler, governance log, backup/recovery — no external MLOps framework), the
entire `src/evaluation/` framework (metrics, golden-dataset runner, robustness harness, experiment
tracker), and the frontend page (plain HTML/CSS/JS, no framework).

**Third-party** — libraries: FastAPI, Uvicorn, Pydantic, `sentence-transformers`, `pypdf`,
`python-docx`, `boto3`/`botocore`, `openai` (SDK, used against NVIDIA NIM/formerly GitHub Models),
`PyJWT`, Microsoft Presidio + spaCy, OpenTelemetry, `pytest`, `uv`. Pretrained models (downloaded,
not trained here): `BAAI/bge-base-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`,
`cross-encoder/nli-deberta-v3-base`, `en_core_web_sm`. External AI APIs: AWS Bedrock (Claude
Haiku 4.5), NVIDIA NIM, Jina AI, Cohere. Managed AWS services (infrastructure, not code):
OpenSearch Service, S3, SQS, Cognito, ECS Fargate (Express Mode), ECR, IAM, Secrets Manager,
Bedrock, ELB/ALB, CloudWatch Logs, CloudTrail. CI/CD: GitHub Actions plus its marketplace actions
(`actions/checkout`, `aws-actions/configure-aws-credentials`, `aws-actions/amazon-ecr-login`,
`docker/build-push-action`, `aws-actions/amazon-ecs-deploy-express-service`) — deployed via
`.github/workflows/deploy-aws.yml` using GitHub OIDC → AWS IAM (no long-lived AWS keys in GitHub).

In short: the pipeline architecture, orchestration, guardrail logic, MLOps backbone, evaluation
framework, and API/frontend are custom; the ML models, cloud infrastructure, and generic
libraries they run on are third-party.

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

# run robustness evaluation (unanswerable/adversarial queries)
uv run python evaluation/run_robustness_eval.py --dataset evaluation/robustness_dataset.json

# verify RetrievalRelevanceGuard's calibrated thresholds against a real dense embedder
uv run python scripts/retrieval_relevance_guard_verification.py
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

**Request flow:** `app/main.py` (FastAPI routes `/ingest`, `/documents`, `/documents/{id}`,
`/documents/reindex`, `/documents/jobs/{id}`, `/ask`, `/ask/debug`, `/health`, `/ready`, plus
`/admin/*`) → `RAGService` (`app/services/rag_service.py`), which owns the whole pipeline:

1. **Ingestion** (`src/ingestion/`) — `IngestionPipeline.ingest_file()` picks a parser via
   `ParserFactory` (dispatches on file extension: `.pdf`/`.docx`/`.md`/`.markdown`), parses to a
   `Document`, then normalizes whitespace via `TextCleaner`. Every parser and pipeline stage
   returns a `Result[T]` (success/data/error) instead of raising — errors carry a `code` +
   `message` and are surfaced up through `RAGService.ingest()` as strings in
   `IngestResponse.errors`, never as exceptions. `RAGService.ingest_allowed_dir` (optional,
   `None` = unrestricted) gates which paths `ingest()` will even hand to the pipeline - resolves
   symlinks/`..` segments and requires the result to sit inside that directory, rejecting anything
   outside with a `PATH_NOT_ALLOWED` error instead of reading it. `None` by default (direct
   construction - tests, `main.py`'s demo run - stays unrestricted); `service_factory
   .build_rag_service()` always sets it from `INGEST_ALLOWED_DIR` (default `sample_documents`) for
   the live API, since `/ingest` takes attacker-controlled paths over an unauthenticated HTTP
   endpoint and previously had zero path validation - `{"file_paths": ["CLAUDE.md"]}` or any other
   `.pdf`/`.docx`/`.md` file reachable inside the container would get indexed and readable back via
   `/ask`. `RAGService.index_document(document)` factors the chunk+embed+index half of `ingest()`
   out into its own method (shared by the synchronous path above and the async worker below), and
   `delete_document(document_id)` / `reindex_document(file_path)` give the document lifecycle a
   real delete and a delete-then-reingest replace (chunk boundaries can shift on any content
   change, so this is never a partial update) - both live-verified end to end against a real
   OpenSearch domain (upload, delete confirmed via a direct index count, reindex confirmed
   non-duplicating).

   **Async ingestion** (`src/ingestion/s3_document_store.py`, `sqs_ingestion_worker.py`,
   `src/mlops/ingestion_job_store.py`) - an alternative entry point for large/slow documents over
   HTTP without blocking the request: `POST /documents` (multipart upload) validates the file via
   `S3DocumentStore.validate()` (extension allowlist, `S3_MAX_FILE_SIZE_MB`), uploads it to
   `S3_BUCKET` under `S3_RAW_PREFIX`, creates a job record via `IngestionJobStore` (S3-backed,
   `jobs/` prefix by default - deliberately not a new database, matching the project's own
   "don't introduce a database without a concrete requirement" rule), and enqueues a message onto
   `SQS_QUEUE_URL` when async ingestion is on. `SQSIngestionWorker.poll_once()` (long-polled,
   `wait_time_seconds`) is driven by `main.py`'s `_sqs_ingestion_loop()` the same way the
   `Scheduler` is - an `asyncio` task in `lifespan`, no separate worker process - and for each
   message: checks `IngestionJobStore.is_already_processed()` first (idempotent redrive/DLQ
   retries), downloads via `S3DocumentStore.download_to_temp()`, calls
   `IngestionPipeline.ingest_from_s3()` (parses the temp file then overrides `document_id`/
   `source`/`file_name` with the real S3 key, since the parser only sees a meaningless temp
   filename), indexes via `RAGService.index_document()`, transitions the job
   RECEIVED→PROCESSING→INDEXED/FAILED, and deletes the SQS message only on success or a
   detected-duplicate skip - never on failure, so SQS's own redrive policy/DLQ handles retries
   rather than the worker inventing its own retry logic. `GET /documents/jobs/{job_id}` polls job
   status. Live-verified end to end against real S3 + SQS + OpenSearch (upload → S3 → SQS → worker
   → indexed → retrievable via `/ask`, groundedness 0.93 on the round trip).
2. **Chunking** (`src/rag/chunking/recursive_chunker.py`) — `RecursiveChunker` splits a
   `Document` into `Chunk`s: first by heading-like lines (`_looks_like_heading`), then within each
   section by sentence boundaries up to `chunk_size` with `chunk_overlap`, merging any resulting
   chunk under `minimum_chunk_size` into its neighbor. Chunk metadata carries the parent section
   title and document metadata forward. `_split_sections` only closes a section once it has real
   body content beyond just the heading line itself - a run of consecutive heading-like lines (a
   table of contents, repeated running headers/footers from PDF extraction) keeps accumulating
   into the same pending section instead of each becoming its own near-empty chunk. Without this,
   every TOC entry ("Attributes of the AI RMF 3", page number and all) becomes a standalone
   one-line chunk that can outrank real content on an exact-phrase query, since it IS that exact
   phrase - a real, previously-shipped bug that was silently gutting retrieval quality on
   `sample_documents/AI-RMF-1stdraft.pdf` (209 chunks before the fix, 148 after, `evaluation/
   golden_dataset.json` was rebuilt against the corrected output).
3. **Embedding** (`src/rag/embeddings/`) — `Embedder` protocol. `SentenceTransformerEmbedder`
   (real local model, default `BAAI/bge-base-en-v1.5`, configurable via `EMBEDDING_MODEL_NAME`) is
   the default (`EMBEDDING_PROVIDER=sentence_transformer`) - a deterministic hash-based embedding
   is not good enough to be what the live app silently falls back to. `HashingEmbedder` (no
   external model/credentials, zero network dependency) is still available via
   `EMBEDDING_PROVIDER=hashing`, and remains `RAGService.__init__`'s own bare-construction default
   (direct `RAGService()` calls - tests, scripts) precisely so the test suite stays fast and
   offline; `tests/unit/conftest.py` globally patches `SentenceTransformer` with a fake that
   delegates to `HashingEmbedder` internally, so anything going through
   `service_factory.build_rag_service()` in a test - including `app.main`'s module-level call -
   never downloads a real model either, while still getting deterministic, content-sensitive
   embeddings for retrieval-ranking assertions to depend on. `main.py`'s demo script goes through
   `build_rag_service()` rather than a bare `RAGService()` for this same reason - it should show
   the same embedding quality as the deployed app, not the offline fallback.
   `SentenceTransformerEmbedder` loads the model once at construction and reuses it for every
   call. `service_factory.build_rag_service()` builds this embedder once and shares the same
   instance between retrieval and `HallucinationDetector` (when enabled), rather than each
   independently constructing its own - keeps
   groundedness scoring consistent with whatever embedding space retrieval is actually using.
   `JinaEmbedder`/`CohereEmbedder` (`EMBEDDING_PROVIDER=jina`/`cohere`, `rag/embeddings/
   jina_embedder.py`/`cohere_embedder.py`) are API-based alternatives with real retry/backoff
   (`EMBEDDING_TIMEOUT_SECONDS`/`EMBEDDING_MAX_RETRIES`) - live-verified against the real Jina API
   via `scripts/jina_live_verification.py` (a real `embed_batch()` call returned two genuine
   1024-dim vectors with real token usage reported back by the API); Cohere is unit-tested only,
   not live-verified (no funded key available). `Embedder.dimensions`/`.provider_name`/
   `.embed_batch()` are part of the Protocol so every implementation (including `HashingEmbedder`)
   exposes the same shape.
   `sentence_transformer` staying the app-level `Settings` default is deliberately scoped to local
   development and bare/no-env-var construction (tests, scripts, a laptop with no cloud
   credentials) - it is *not* what the AWS deployment actually runs. `terraform/ecs.tf` overrides
   this to `EMBEDDING_PROVIDER=jina` explicitly for the live ECS task, since the platform spec
   requires the AWS deployment to be API-first rather than downloading a model into the task (see
   the Reranking section below for the identical `RERANKER_PROVIDER` override, and
   `terraform/README.md`'s cost-summary note on the resulting per-call API cost).
4. **Vector store** (`src/rag/vector_store/`) — `VectorStore` protocol. `InMemoryVectorStore` does
   brute-force cosine similarity over an in-process dict, used for local/dev/tests - this stays the
   default for direct `RAGService()` construction. `OpenSearchVectorStore` is the production
   adapter and is now real, config-driven wiring (`VECTOR_STORE_PROVIDER=opensearch`), not just an
   available-but-unreachable class: `service_factory._build_vector_store()` builds an
   AWS-SigV4-signed client via `opensearch_client_factory.build_opensearch_client()` (ambient
   boto3 credentials, never a stored username/password) and calls `store.ensure_index()` once at
   startup so a fresh domain gets the correct `knn_vector` mapping without a manual provisioning
   step. `OpenSearchVectorStore` itself now supports real bulk indexing (`add_many` via the
   `_bulk` API instead of one request per chunk), a genuine BM25 lexical search
   (`search_lexical()`, a real OpenSearch `match` query - not the regex-overlap approximation
   `HybridRetriever` still uses for the in-memory path), `delete()`/`delete_by_document()` (no
   orphaned chunks when a document is removed), `update_metadata()` (partial update, no
   re-embedding needed), and `health_check()`. Embedding dimension is validated against the
   index's mapping on every `add`/`add_many`/`search` when `embedding_dimensions` is set (always
   true for the live-wired path, since it's threaded through from the active `Embedder`).
5. **Retrieval** (`src/rag/retrieval/`) — `HybridRetriever` runs real, independent dense and BM25
   searches and fuses them with Reciprocal Rank Fusion (RRF), not a linear blend of raw scores
   from two incomparable scales. `dense_top_k`/`bm25_top_k` (default 20 each, env
   `DENSE_TOP_K`/`BM25_TOP_K`) control how deep each individual search goes; `rrf_k` (default 60,
   env `RRF_K`) is the standard RRF damping constant (`1 / (rrf_k + rank)` per list, summed across
   lists a chunk appears in). `bm25.py` implements real Okapi BM25 (TF/IDF/length normalization,
   not word-overlap) - used by `InMemoryVectorStore.search_lexical()` so local/test retrieval
   scores lexical relevance the same way the OpenSearch-backed path's real `search_lexical()`
   does, rather than two different algorithms silently diverging between dev and prod. Every
   `RetrievedChunk` now carries `retrieval_method` (`"dense"` / `"bm25"` / `"both"`, telling you
   which method(s) actually surfaced this chunk) and `rank` (1-indexed position, reassigned by the
   reranker when one runs) - both flow through generation and guardrails untouched, and both feed
   the retrieval-debugging trace below. `Chunk` also carries lineage metadata (`content_hash`,
   `chunking_version`, `embedding_provider`/`model`/`version`, `document_version`, `indexed_at`)
   for detecting stale/incompatible chunks across a config change, and document-level
   authorization fields (`tenant_id`, `access_groups`, `classification`) - `RAGService
   ._filter_by_access()` enforces the one rule that's actually opinionated here: a chunk with a
   non-empty `access_groups` is only returned when the caller's own groups (from
   `AuthenticatedUser.claims["access_groups"]`) intersect it, applied *before* the reranker/prompt
   ever see the chunk, never retrieve-then-hide. `OpenSearchIndexManager`
   (`opensearch_index_manager.py`) implements a versioned-index + alias pattern
   (`{base_name}-v1`, `-v2`, ...) with atomic `_aliases` repoint (`switch_alias`) and `rollback` -
   zero-downtime reindexing without a manual cutover script. Live-verified against a real
   OpenSearch domain (version creation, listing, alias switch, rollback all confirmed).

   **Retrieval debugging trace** (`rag/retrieval/trace.py`, `RAGService.ask_with_trace()`) - the
   normal `ask()` path stays exactly as it was (no tracing overhead for regular callers);
   `ask_with_trace()` is a parallel method that runs the identical pipeline while also recording a
   `RetrievalTrace`: raw dense and BM25 candidate lists (not just the fused result), the fused RRF
   ranking, reranked candidates (when a reranker ran), final chunk ids, generation provider,
   groundedness, guardrail findings, and per-stage latency in milliseconds (`embedding`,
   `dense_search`, `bm25_search`, `rrf_fusion`, `rerank`, `generation`, `output_guardrails`,
   `total`). Exposed via `POST /ask/debug`, gated behind the `DEBUG_QUERY` permission (granted to
   `ML_ENGINEER`/`DATA_SCIENTIST`/`ADMINISTRATOR`, not `READ_ONLY`/`REVIEWER`) since it surfaces
   internal scoring detail regular `QUERY`-only callers shouldn't see.
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
   `JinaReranker`/`CohereReranker` (`RERANKER_PROVIDER=jina`/`cohere`, `rag/retrieval/
   jina_reranker.py`/`cohere_reranker.py`) are API-based alternatives with the same
   `rerank(query, candidates, top_k)` shape - live-verified against the real Jina API (correctly
   ranked a relevant chunk at 0.7295 vs an irrelevant one at 0.0284, via
   `scripts/jina_live_verification.py`); Cohere is unit-tested only, not live-verified (no funded
   key available). Same split as embeddings above: `local` (the `CrossEncoderReranker` cross-
   encoder) stays the app-level default for local dev/tests/bare construction, but
   `terraform/ecs.tf` overrides this to `RERANKER_PROVIDER=jina` for the live ECS task - the AWS
   deployment doesn't download this model either.
7. **Generation** (`src/rag/generation/`) — `Answerer` protocol. `ExtractiveAnswerer` picks the
   best-overlap sentence from retrieved chunks (no LLM call, fully deterministic — grounded by
   construction since it only ever returns retrieved text). `BedrockAnswerer` and
   `OpenAICompatibleAnswerer` are LLM-backed adapters that share one grounded prompt template
   (`rag/generation/prompt.py::build_grounded_prompt`) so every provider answers only from
   retrieved context and cites the same source chunk ids. `BedrockAnswerer` takes an injected
   boto3 `bedrock-runtime` client and calls the Converse API (`client.converse`) rather than
   `invoke_model` with a provider-specific JSON body - Converse works uniformly across every
   Bedrock model provider and across inference profile ARNs (some newer Claude models, e.g.
   Claude Haiku 4.5, are only invocable via an inference profile rather than a plain model ID;
   `model_id` can be either shape without this class caring which). `OpenAICompatibleAnswerer`
   builds its own `openai.OpenAI`
   client from `api_key`/`base_url`/`model_name` and works against any OpenAI-compatible Chat
   Completions endpoint (OpenAI, Azure OpenAI, GitHub Models, Ollama, OpenRouter, Groq, ...) by
   changing only those config values — no code changes. It returns a fixed fallback string
   without calling the LLM when there are no retrieved chunks, retries only transient failures
   (HTTP 429/500/502/503/504) with exponential backoff, and returns a fallback string on
   exhausted/non-retryable failures instead of raising. `FallbackAnswerer` wraps any two
   `Answerer` instances - tries the primary, and on *any* exception from it (Bedrock throttling,
   an AWS Marketplace billing/subscription error, a network failure, whatever) falls back to the
   secondary rather than surfacing a 500 to the caller. It's generic composition over the
   `Answerer` Protocol with no knowledge of which concrete providers it wraps. Wired via
   `GENERATION_FALLBACK_PROVIDER` (same allowed values as `GENERATION_PROVIDER`) - when set,
   `service_factory.build_rag_service()` builds a second answerer for that value and wraps the
   primary in `FallbackAnswerer`; when unset (the default), `RAGService.answerer` is just the
   primary provider, unchanged from before this existed.

   **Citation validation** (`rag/generation/citations.py`) - `build_grounded_prompt` instructs
   every LLM-backed provider to cite claims inline as `[Source N]`, matching that source's number
   in the prompt's own numbered context list. `extract_citations(answer, retrieved_chunks)` parses
   those markers back out of the generated answer and resolves each one against the real numbered
   source list, distinguishing a genuinely different failure mode from `HallucinationDetector`'s
   whole-answer overlap score: a *specific* fabricated claim of provenance (citing "[Source 9]"
   when only 3 sources were ever provided) rather than a generic groundedness shortfall.
   `AskResponse.citations` carries one `CitationResponse` per marker found (document/version/
   chunk/section for valid ones, just the number for invalid ones); `guardrail_flags
   .has_invalid_citations` is set whenever any citation resolves to `valid: false`.
   `ExtractiveAnswerer` output correctly yields an empty citations list (it copies chunk text
   verbatim rather than generating citation markers) - this is expected behavior, not a gap.
   Citations are extracted from the *final* answer text (after any abstention replacement), so an
   abstained response never carries stale citations from the discarded original answer.

   **Cost observability** (`rag/generation/telemetry.py::record_generation`) - `BedrockAnswerer`
   and `OpenAICompatibleAnswerer` both call this right after a successful response, reading real
   token usage straight from the provider's own response (`response["usage"]` for Bedrock's
   Converse API, `response.usage.prompt_tokens`/`.completion_tokens` for OpenAI-compatible Chat
   Completions) - never estimated by counting characters. Deliberately *not* stored as instance
   state on the Answerer (these are shared singletons across concurrent requests, built once at
   startup - a "last usage" attribute would be a real race condition), just recorded straight into
   OTel counters (`generation.requests`, `generation.input_tokens`, `generation.output_tokens`,
   `generation.estimated_cost_usd`) the same way `rag/guardrails/telemetry.py` already does -
   flows through the CloudWatch EMF exporter above for free, zero additional wiring.
   `MODEL_COST_PER_1K_TOKENS` is a short, explicit, hand-maintained price table (not a pricing API
   integration - "keep it lightweight" is the point) with Bedrock inference-profile ARNs resolved
   to their trailing model-id segment so one entry covers every region/account. A model not in the
   table gets `cost=None`, not a silently wrong zero - `estimate_cost_usd()` and `record_generation`
   both treat "unknown" as a distinct, honestly-reported case. `ExtractiveAnswerer` records nothing
   here (no LLM call, no tokens, no cost - correctly nothing to report).
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
   findings show up automatically without a schema change. `GuardrailManager.default()` itself
   (used only by bare `RAGService()` construction - tests/scripts) still wires only those two;
   three more lightweight, dependency-free guardrails exist and implement the same interface but
   need explicit registration to run under `.default()`: `PromptInjectionGuard` (input stage,
   regex heuristics in `injection_patterns.py` for injection/jailbreak phrasing - "ignore previous
   instructions", "you are now in developer mode", "reveal your system prompt", DAN-style
   jailbreaks, secret-exfiltration requests), `SecretLeakageGuard` (output stage, API key/token/
   private-key patterns), `ProfanityGuard` (output stage, small illustrative wordlist).
   `IndirectPromptInjectionGuard` (`indirect_prompt_injection_guard.py`, output stage) scans
   `context.retrieved_chunks` text with the same pattern set - a fundamentally different attack
   surface than the input-stage guard (malicious text embedded in a *retrieved document*, not the
   user's query), paired with `rag/generation/prompt.py::build_grounded_prompt`'s explicit
   `SYSTEM_FRAMING` instructing the model to treat retrieved context as untrusted evidence, not
   instructions - defense-in-depth, not a claimed-complete solution. For the live app,
   `service_factory._build_guardrail_manager()` wires `PromptInjectionGuard` and
   `IndirectPromptInjectionGuard` first, **on by default**
   (`PROMPT_INJECTION_GUARD_ENABLED`/`INDIRECT_PROMPT_INJECTION_GUARD_ENABLED`, both default
   `true`), ahead of the Phase 1 pair - live-verified end to end against a real Cognito-
   authenticated request: four distinct adversarial phrasings (direct injection, jailbreak
   persona, fake system-message override, developer-mode + secret exfiltration) all correctly
   blocked (`evaluation/robustness_dataset.json`, `tests/unit/test_robustness_eval.py`).

   **Abstention** (`RAGService.ask()`, `ABSTENTION_MESSAGE`, `abstention_enabled` - default
   `true`, `ABSTENTION_ENABLED`) - `RAGService._should_abstain()` replaces the user-facing
   `answer` text with an honest "I don't have enough supporting evidence..." message (sources kept
   for auditability) when either of two independent, complementary guardrail signals fires -
   they catch different failure modes, not the same one twice:
   - `hallucination: true` (`HallucinationDetector`, on by default) - does the answer match its
     own retrieved evidence. Still only a `WARN`, never an auto-`BLOCK`.
   - `low_retrieval_relevance: true` (`RetrievalRelevanceGuard`, **off by default** - see below).

   `AskResponse.confidence` is derived from `groundedness`, never the raw retrieval/rerank score -
   conflating the two was a real bug fixed this session, since a perfect retrieval match can still
   pair with a fabricated answer.

   **Retrieval-relevance guard** (`rag/guardrails/retrieval_relevance_guard.py`,
   `RETRIEVAL_RELEVANCE_GUARD_ENABLED` - default `false`, `RETRIEVAL_RELEVANCE_THRESHOLD` optional
   override) - fixes a real, verified gap that groundedness alone can't catch: it measures whether
   the answer matches its own retrieved chunks, not whether those chunks are actually relevant to
   the query. `ExtractiveAnswerer`'s "answer" is always copied verbatim from a chunk, so it's
   tautologically high-groundedness even for a completely off-topic query against fully irrelevant
   retrieved content (found by `tests/unit/test_robustness_eval.py` against the real PDF: ~0.85-0.90
   groundedness for a query about the boiling point of mercury). `RetrievalRelevanceGuard` embeds
   the query and the top retrieved chunk(s) directly through the shared `Embedder` and compares
   cosine similarity against an embedder-appropriate threshold
   (`default_retrieval_relevance_threshold()`) - deliberately *not* the RRF-fused retrieval score
   (rank-based, not magnitude-based - a top-1-in-both-lists chunk scores nearly the same whether
   the match is great or terrible) or the vector store's own raw score (not comparable across
   backends - OpenSearch's k-NN score formula and `InMemoryVectorStore`'s raw cosine aren't on the
   same scale). Two named threshold defaults, not one universal number, verified by
   `scripts/retrieval_relevance_guard_verification.py` against `evaluation/golden_dataset.json`'s
   24 real queries through the actual `RAGService._retrieve()` pipeline (not an idealized
   full-corpus scan): with a genuine dense embedder (`BAAI/bge-small-en-v1.5`), threshold `0.68`
   gives **zero false positives** across all 24 real answerable queries and catches 3 of 4 known
   unanswerable cases. `HashingEmbedder`'s crude vectors don't separate reliably enough for this
   signal at all - real answerable queries score as low as 0.29, overlapping with unanswerable
   queries up to 0.43 - so its default threshold (`0.20`) is deliberately set to be a safe no-op
   (zero false positives, zero catches) rather than a falsely-confident number. This is exactly why
   the guard defaults to **disabled everywhere** (`GuardrailManager.default()`,
   `service_factory`) unlike `PIIGuard`/`HallucinationDetector` - and why
   `tests/unit/test_robustness_eval.py`'s known-gap test still shows the gap: the whole test suite
   runs on hash-quality embeddings (`tests/unit/conftest.py`'s `SentenceTransformer` mock delegates
   to `HashingEmbedder` for speed), so the real fix can only be demonstrated with genuine dense
   embeddings, via the verification script, outside the fast/offline unit-test path. Re-run that
   script and update `DENSE_EMBEDDER_DEFAULT_THRESHOLD` if `EMBEDDING_MODEL_NAME` changes to a
   materially different model.

   **A third, Jina-specific threshold** (`JINA_EMBEDDER_DEFAULT_THRESHOLD`, `0.30`) exists because
   this exact gap was hit for real in the live AWS deployment, twice: `EMBEDDING_PROVIDER=jina`
   became the AWS default (see the Embedding section above) without ever re-running the calibration
   script against Jina's actual embedding space - the code just fell through to
   `DENSE_EMBEDDER_DEFAULT_THRESHOLD` (0.68, calibrated for `BAAI/bge-small-en-v1.5`), on the
   unverified assumption that one dense embedder's threshold would transfer to another. It didn't:
   a genuinely on-topic, answerable question ("What are the trustworthiness characteristics of AI
   systems?") scored 0.60 in production and incorrectly abstained. Re-running
   `scripts/retrieval_relevance_guard_verification_jina.py` (same methodology, real Jina API calls)
   against the real 24-query golden dataset found the deployed 0.68 threshold produces **14/24
   false positives** against Jina's embedding space - unusable; `0.37` (just under that dataset's
   observed answerable minimum of 0.377) restored zero false positives against it, still catching 2
   of 4 unanswerable cases.

   That single-document calibration didn't generalize. Live-testing against a second, real,
   different document (an unrelated HR handbook, never part of any golden dataset) found a
   genuinely answerable question scoring 0.36 - just under 0.37, and *below* the AI-RMF
   calibration's own observed minimum, sitting only ~0.02 away from two of the four original
   unanswerable scores (0.333, 0.348). A cosine-similarity cutoff tuned on one document's 24
   queries doesn't reliably hold on a different document's content - the real separation in Jina's
   embedding space between "clearly unrelated" and "genuinely relevant" is narrower and more
   document-dependent than the first calibration run could show. `0.30` restores real margin below
   both data points, at the honest cost of no longer reliably catching any of the four original
   unanswerable queries - an acceptable trade, since false-positive abstention on real content is a
   worse failure than an occasional missed catch, and `HallucinationDetector`'s independent
   groundedness check still guards against ungrounded answers regardless. A properly generalizing
   fix would calibrate against a golden dataset spanning multiple real documents, not one - not
   done here; treat any single-document calibration for this guard as provisional.
   `default_retrieval_relevance_threshold()` dispatches on `embedder.provider_name == "jina"` for
   this. This bug and its fix (twice) are the concrete argument for the general rule stated above -
   re-verify per provider, never assume a threshold transfers. `PolicyEngine`
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

   **CloudWatch metrics** (`app/observability.py::CloudWatchEMFMetricExporter`,
   `CLOUDWATCH_METRICS_ENABLED` - default `false`) - the concrete exporter that plugs into the
   `MeterProvider` extension point above, specifically for a plain ECS Fargate task with no
   sidecar/collector. Writes CloudWatch Embedded Metric Format (EMF) JSON lines to a dedicated
   `enterprise_rag_platform.cloudwatch_emf` logger (`propagate=False`, its own `StreamHandler` to
   stdout so the raw JSON line is never mixed with the app's normal formatted logs) - CloudWatch
   Logs auto-detects the `_aws` EMF structure in any log entry sent through the log group the ECS
   task already writes to (via the `awslogs` driver already in the task definition) and creates/
   updates the named metrics automatically, no extra IAM permission beyond
   `logs:PutLogEvents` the execution role already has, no ADOT collector. Wired in `main.py` at
   module level, before `build_platform_manager()`/`build_rag_service()` run, via
   `PeriodicExportingMetricReader(CloudWatchEMFMetricExporter(...), export_interval_millis=...)`
   (`CLOUDWATCH_METRICS_NAMESPACE` default `EnterpriseRAGPlatform`,
   `CLOUDWATCH_METRICS_EXPORT_INTERVAL_SECONDS` default `60`). `Sum` instruments export as their
   raw counter value; `Histogram` instruments (`guardrail.latency`, `guardrail.groundedness_score`)
   export as `{name}.avg` (mean = sum/count) and `{name}.count` per interval - a deliberate
   simplification, not full bucket/percentile fidelity, since "is average latency creeping up"
   is the signal this project actually needs rather than percentile-accurate histograms. Verified
   against real OTel `MetricsData` produced through the same shared `TELEMETRY_READER` every other
   telemetry test uses (`tests/unit/test_cloudwatch_emf_exporter.py`) - real EMF JSON structure,
   correct `Namespace`/`Dimensions`/`Metrics` envelope, correct histogram mean/count split. Off by
   default like every other opt-in provider flag in this file; CloudWatch does charge per custom
   metric name past the first 10/month, so enabling this in a real deployment is a real (small)
   cost decision, not a free one.

**Authentication & authorization** (`app/auth.py`, `mlops/permissions.py`) - generic,
vendor-neutral OIDC/JWT, not tied to any specific identity provider: `OIDCTokenValidator(issuer,
audience, jwks_url, role_claim="role", default_role=Role.READ_ONLY)` uses `PyJWKClient` + PyJWT,
pins `algorithms=["RS256"]` explicitly (rejects `alg: none` attacks), and validates
issuer/audience/expiry/signature against the IdP's real JWKS endpoint. `AUTH_ENABLED` (default
`false`) is the master switch; when `true`, `main.py`'s `get_current_user()` dependency requires a
`Bearer` token on every RBAC-gated route and 401s on a missing/malformed header or a validation
failure, 403s via `require_permission(permission)` when the token's role lacks it. Live-verified
end to end this session against a real AWS Cognito User Pool (not synthetic RSA test tokens): a
real Cognito-issued ID token validated correctly against `OIDCTokenValidator` with the pool's real
JWKS, and separately through the running app with `AUTH_ENABLED=true` - no token → 401, a
signature-tampered token → 401, a valid real token → 200 with the full guardrail pipeline running.
RBAC (`mlops/permissions.py::ROLE_PERMISSIONS`, a static `Role` × `Permission` matrix) covers both
the pre-existing MLOps-platform permissions and RAG API-layer ones: `QUERY` (every role),
`UPLOAD_DOCUMENT`/`DELETE_DOCUMENT` (`ML_ENGINEER`, `DATA_SCIENTIST` upload-only,
`ADMINISTRATOR`), and `DEBUG_QUERY` (`ML_ENGINEER`/`DATA_SCIENTIST`/`ADMINISTRATOR`, gated from
`READ_ONLY`/`REVIEWER` since it exposes internal retrieval-trace detail). `access_groups` on
`AskRequest`/`RAGService.ask()` come only from the validated token's own claims, never the
request body - a caller cannot claim arbitrary access groups for themselves.

**Terraform wiring for auth** - `terraform/ecs.tf` used to never set `AUTH_ENABLED`/`OIDC_*` at
all, so the Terraform-deployed ECS task always ran with authentication off regardless of what the
app itself supports, and `existing_cognito_user_pool_id` sat declared but unreferenced by any
`data`/`resource` block - a real gap between "the feature is built and live-verified" and "the
deployment path actually turns it on." Fixed: `terraform/data.tf` now has a real
`data "aws_cognito_user_pool" "existing"` (reuses the real pool, `us-east-1_jkzIa7abx` by default -
never recreates it, since a new pool means a new pool ID and every already-configured `OIDC_*`
value elsewhere goes stale), and `ecs.tf` sets `AUTH_ENABLED=true` plus `OIDC_ISSUER`/
`OIDC_JWKS_URL`/`OIDC_AUDIENCE` (the last from the new `cognito_app_client_id` variable) whenever
that pool reference resolves. Blanking `existing_cognito_user_pool_id` disables auth entirely
rather than triggering fresh-pool creation - this module has never actually implemented that,
despite the variable's old description implying otherwise.

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
`AWS_REGION`, `BEDROCK_MODEL_ID`, `RERANKER_*`, etc.). `GENERATION_PROVIDER` accepts `extractive`
(default), `openai_compatible` (requires `LLM_BASE_URL` + `LLM_API_KEY`, raises
`ServiceConfigurationError` if either is missing), or `bedrock` (builds a real
`boto3.client("bedrock-runtime", region_name=settings.aws_region)` and constructs
`BedrockAnswerer` with it — no injected client needed, since `boto3` resolves credentials
automatically from whatever IAM role/identity the process is running as, e.g. an ECS task role);
any other value raises `ServiceConfigurationError`. `VECTOR_STORE_PROVIDER` accepts `memory`
(default) or `opensearch` - the latter builds its own authenticated client via
`opensearch_client_factory.build_opensearch_client()` (ambient boto3 credentials sign every
request, same pattern as `BedrockAnswerer`) and requires `OPENSEARCH_HOST` to be set
(`ServiceConfigurationError` otherwise); `OPENSEARCH_INDEX` (default `enterprise-rag-chunks`),
`OPENSEARCH_PORT` (default `443`), `OPENSEARCH_USE_SSL`/`OPENSEARCH_VERIFY_CERTS` (default
`true`), `OPENSEARCH_CONNECT_TIMEOUT` (default `5` seconds), and `OPENSEARCH_MAX_RETRIES`
(default `3`) tune the connection. `AWS_REGION` is reused for SigV4 signing - no separate
OpenSearch-specific region setting.
`EMBEDDING_PROVIDER` accepts `sentence_transformer` (default; model name via
`EMBEDDING_MODEL_NAME`, default `BAAI/bge-base-en-v1.5`) or `hashing` - both are wired, no
injected client needed since sentence-transformers loads the model itself from HuggingFace on
first use.
`RERANKER_ENABLED` (default `true`),
`RERANKER_MODEL_NAME`, and
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
`RERANKER_PROVIDER` accepts `local` (default, `CrossEncoderReranker`), `jina` (requires
`JINA_API_KEY`), or `cohere` (requires `COHERE_API_KEY`) - same
`ServiceConfigurationError`-if-misconfigured pattern as generation/vector-store providers.
`EMBEDDING_PROVIDER` additionally accepts `jina`/`cohere` (`JINA_API_KEY`/`JINA_EMBEDDING_MODEL`/
`JINA_EMBEDDING_DIMENSIONS`, `COHERE_API_KEY`/`COHERE_EMBEDDING_MODEL`/
`COHERE_EMBEDDING_DIMENSIONS`; both share `EMBEDDING_TIMEOUT_SECONDS`/`EMBEDDING_MAX_RETRIES`).
`S3_BUCKET` (unset by default - async ingestion endpoints/worker only build when set) with
`S3_RAW_PREFIX`/`S3_PROCESSED_PREFIX`/`S3_FAILED_PREFIX` (default `raw/`/`processed/`/`failed/`),
`S3_MAX_FILE_SIZE_MB` (default `25`), `S3_JOBS_PREFIX` (default `jobs/`) configure
`S3DocumentStore`/`IngestionJobStore`. `SQS_QUEUE_URL` (unset by default) plus
`ASYNC_INGESTION_ENABLED` (default `false`) together gate whether `main.py` builds and runs
`SQSIngestionWorker`'s polling loop; `SQS_POLL_INTERVAL_SECONDS` (default `20`) controls how often
`_sqs_ingestion_loop()` calls `poll_once()`. `AUTH_ENABLED` (default `false`) plus `OIDC_ISSUER`/
`OIDC_AUDIENCE`/`OIDC_JWKS_URL`/`OIDC_ROLE_CLAIM` (default role-claim name `role`) configure
`OIDCTokenValidator` - all three of issuer/audience/jwks_url are required when auth is enabled.
`ABSTENTION_ENABLED` (default `true`) controls the low-groundedness abstention replacement
independently of `HALLUCINATION_GUARD_ENABLED` itself (the guard can run and flag without
abstention rewriting the answer text, if a caller wants the raw flag without the behavior change).

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
`Scheduler.trigger()`), `GET /admin/backups` (`TRIGGER_BACKUP` permission, lists snapshot ids via
`PlatformManager.list_backups()` - the durable target's ids when one is configured, local snapshot
files otherwise), `POST /admin/backups/restore` (`TRIGGER_RESTORE`, `ADMINISTRATOR`-only per
`ROLE_PERMISSIONS` - restoring silently overwrites current platform state; body `{"snapshot_id":
...}`, always restores from the durable target via `restore_backup_from_target()`, `400` when no
target is configured rather than `404`, since that's a config problem not a missing-snapshot one).
This closes a real gap: the scheduled `backup` job wrote snapshots automatically, but nothing in
the app could ever read one back - `RecoveryManager`/`restore_backup_from_target()` had zero HTTP-
reachable callers before this existed, so a backup nobody could restore was close to worthless
operationally. All six 404 when `platform_manager` is `None`, with a detail message that
distinguishes "MLOPS_ENABLED=false" from an actual startup failure (see below). Document-lifecycle
and debugging endpoints, each RBAC-gated: `POST /ingest` (`UPLOAD_DOCUMENT`, synchronous, local/
allowed-dir file paths), `POST /documents` (`UPLOAD_DOCUMENT`, multipart upload, async via S3/SQS
when configured), `GET /documents/jobs/{job_id}` (`QUERY`), `DELETE /documents/{document_id}`
(`DELETE_DOCUMENT`), `POST /documents/reindex` (`UPLOAD_DOCUMENT`, body `{"file_path": ...}`),
`POST /ask` (`QUERY`), `POST /ask/debug` (`DEBUG_QUERY`, same request shape as `/ask`, returns
`{"response": AskResponse, "trace": RetrievalTraceResponse}`). All document-lifecycle endpoints
plus async ingestion were live-verified end to end against real S3/SQS/OpenSearch this session
(see the Ingestion section above).

**API hardening** (`app/rate_limiter.py`, `main.py`'s upload streaming) - two real gaps closed
together since both sit on the unauthenticated-by-default surface (`AUTH_ENABLED` defaults to
`false`). First: `POST /documents` used to read the *entire* upload into memory
(`f.write(await file.read())`) before `S3DocumentStore.validate()`'s size check ever ran - a
large-enough upload could exhaust process memory before any limit was enforced. Fixed by
streaming the upload in `UPLOAD_CHUNK_SIZE_BYTES` (1 MiB) chunks and aborting with `413` the
moment the running total exceeds `S3DocumentStore.max_file_size_bytes`, before the rest of the
body is ever read or written to disk. Second: nothing rate-limited any endpoint at all.
`rate_limiter.InMemoryRateLimiter` is a small, dependency-free fixed-window limiter (same
"no external store needed at this project's scale" philosophy as the custom BM25/SigV4-client
pieces) wired as ASGI middleware in `main.py`, keyed by client IP (`request.client.host` - runs
before any auth dependency, so it can't key on the authenticated subject) with `/health`/`/ready`
exempted (ECS's own health check polls these continuously; rate-limiting them would make the
health check itself take the service down). `RATE_LIMIT_ENABLED` (default `true`) /
`RATE_LIMIT_REQUESTS_PER_MINUTE` (default `120`) control it. Stated plainly, not hidden: this is
in-process state, so with `ecs_desired_count > 1` each task enforces its own independent window -
the effective global limit across the service is up to N times the configured value, the same
category of caveat the pre-EventBridge interval scheduler had. Correct and sufficient at the
single-task deployment this repo has actually run against AWS; an exact global limit across
multiple tasks would need a shared store (Redis/DynamoDB) instead, not implemented here.

Both `build_platform_manager()`/`build_rag_service()` calls at module level are wrapped in a
`try`/`except Exception` - a model download failing, a Bedrock/network error, anything raised
during construction gets caught, logged, and recorded in module-level `startup_error: str | None`
rather than crashing the whole ASGI process before it can even start. Without this, an unguarded
failure meant `app.main` never finished importing - not even `/health` was reachable, and there
was no way to tell "the app is up but a dependency failed to load" from "the app is completely
dead" (a plausible contributor to some of the harder-to-diagnose ECS deployment failures
encountered before this existed). With it: `/health` returns `503` with the actual exception
message when `startup_error` is set (200 `{"status": "ok"}` otherwise - ECS's health check still
correctly fails and cycles the task, but the container stays reachable and diagnosable while doing
so); `/ingest` and `/ask` return `503` with the same detail instead of raising `AttributeError` on
a `None` `rag_service`; the admin 404s report the real failure reason when there is one, instead
of always claiming `MLOPS_ENABLED=false`.

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
literally how `evaluation/golden_dataset.json` — 24 queries grounded in
`sample_documents/AI-RMF-1stdraft.pdf` at `RecursiveChunker()` defaults — was built; don't
fabricate ids without inspecting real chunks). `category` isn't a fixed enum - besides the
original topic categories (`framework-overview`, `stakeholders`, `trustworthiness`, ...), four
queries cover retrieval-difficulty categories that don't fit the original set:
`semantic-paraphrase` (heavily reworded phrasing of an already-answerable question, same
`relevant_chunk_ids` as its source question - checks the retriever isn't just keyword-matching),
`numeric` (a specific count/fact the corpus states), `comparison` (a question requiring
contrasting two things the corpus discusses together) - each individually verified to retrieve its
relevant chunk within the top 10 before being added, not just passing on aggregate recall.
`tests/unit/test_evaluation_integration.py` runs this dataset through the real PDF end-to-end and
asserts `recall@10 > 0.5` specifically so a future chunker default change that invalidates the ids
fails loudly instead of silently.

**Robustness evaluation (unanswerable/adversarial)** — `src/evaluation/robustness.py`,
`evaluation/robustness_dataset.json`, `evaluation/run_robustness_eval.py`. Neither "unanswerable"
nor "adversarial" queries fit the golden-dataset schema above: `relevant_chunk_ids` must be a
non-empty list (there's structurally no "correct chunk" for a query the corpus doesn't answer),
and an adversarial query is testing guardrail *behavior*, not retrieval recall against golden ids.
This is a parallel, smaller harness that checks real `RAGService.ask()` behavior instead:
`RobustnessCase` (`id`, `query`, `category` - `"unanswerable"` or `"adversarial"`,
`expect_abstention`?, `expect_block`?) and `run_robustness_eval(rag_service, dataset,
abstention_message) -> RobustnessReport` (`pass_rate`, per-case `RobustnessResult` with
`observed_action`: `"block"` / `"abstain"` / `"answered"`, detected the same way
`app.services.rag_service.RAGService.ask()` itself signals a guardrail block - `sources == []` and
`confidence == 0.0` together). `evaluation/robustness_dataset.json` has 8 real cases against
`sample_documents/AI-RMF-1stdraft.pdf`: 4 unanswerable (genuinely off-topic questions like the
boiling point of mercury or a FIFA World Cup result) and 4 adversarial (direct injection, DAN-style
jailbreak, fake system-message override, developer-mode + secret-exfiltration framing).
`tests/unit/test_robustness_eval.py` runs this against the real PDF with a real `RAGService`
(`PromptInjectionGuard` added explicitly, since bare `RAGService()` only wires the Phase 1
defaults - see the Guardrails section above): all 4 adversarial cases reliably block
(**verified**). The 4 unanswerable cases still don't abstain *in this specific test*
(`test_known_gap_with_hash_quality_embeddings_...`) because it runs on `HashingEmbedder` (like the
whole test suite), and `RetrievalRelevanceGuard` - the real fix for this gap, see the Guardrails
section's Abstention/retrieval-relevance-guard entry - is verified safe only with a genuine dense
embedder, not a crude hash-based one. `scripts/retrieval_relevance_guard_verification.py`
demonstrates the actual fix working (0 false positives on 24 real golden queries, 3/4 unanswerable
cases caught) outside the fast/offline unit-test path. `uv run python
evaluation/run_robustness_eval.py --dataset evaluation/robustness_dataset.json` runs the
robustness dataset itself against whatever `GENERATION_PROVIDER`/guardrail config is in the
environment via `service_factory.build_rag_service()` - with `RETRIEVAL_RELEVANCE_GUARD_ENABLED=true`
and a real embedder configured, expect 3/4 unanswerable cases (plus all 4 adversarial) to pass.

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
runs a job immediately, outside its schedule (and is also the entry point the EventBridge-driven
path below uses). Example jobs: re-index documents, run evaluation, health checks, drift
detection, backup - register any zero-argument callable. The FastAPI app is one such "whatever
actually owns scheduling" - see "Wiring" in the main Architecture section for the
`asyncio`-task-in-`lifespan` loop and the two jobs it registers by default.

`_scheduler_loop()`'s plain interval mode has a real bug at `ecs_desired_count > 1`: every ECS
task runs its own `asyncio` loop against its own in-memory `Scheduler`, with zero coordination
between tasks, so a job registered once fires once *per task* every interval - a "backup" job
runs twice per cycle the moment autoscaling brings up a second task (`ecs_max_count` defaults to
2 in `terraform/variables.tf`), not because of any deploy misconfiguration, just because
interval-based scheduling has no single-execution guarantee across replicas.
`mlops/sqs_scheduler_worker.py::SQSSchedulerWorker` fixes this for real by routing job execution
through SQS instead of a per-task clock: `terraform/scheduler.tf` provisions one
`aws_scheduler_schedule` (EventBridge Scheduler) per job, each sending a `{"job_id": ...}` message
to a dedicated SQS queue on its own `rate(N minutes)` schedule; every ECS task polls that same
queue (`_scheduler_sqs_loop()` in `app/main.py`, same `asyncio`-task-in-`lifespan` pattern as the
SQS ingestion worker), but SQS's single-delivery-per-consumer guarantee means only one task
actually receives and processes each message - `SQSSchedulerWorker.poll_once()` calls
`Scheduler.trigger(job_id)` for the job named in the message, then always deletes the message
(including on job failure - the job's own failure is already recorded via `JobRun.success=False`,
and the next EventBridge-scheduled message will fire again on its own cron regardless, so
retrying via SQS redelivery would just duplicate the exact problem this exists to solve).
**Wiring** (`service_factory.build_scheduler_sqs_worker`/`build_scheduler_sqs_client`,
`SCHEDULER_QUEUE_URL` env var, unset by default): unset keeps the plain interval loop exactly as
it was before this existed (correct at `ecs_desired_count = 1`, the only configuration this repo
has actually run against real AWS); once set, `app/main.py`'s `lifespan()` runs the SQS-driven
loop *instead of* the interval loop, never both (running both would double-execute every job,
defeating the point). `SCHEDULER_QUEUE_POLL_INTERVAL_SECONDS` (default `20`, same default as the
ingestion queue's own poll interval) controls how often each task checks the queue - independent
of `SCHEDULER_INTERVAL_SECONDS`, which in this mode only affects `next_run_at` bookkeeping shown
via `/admin/scheduler/jobs`, not when jobs actually run (EventBridge's own cron owns that).
`terraform/scheduler.tf` also creates the one new IAM role this module needed to add
(`scheduler_invocation`, for EventBridge Scheduler's `sqs:SendMessage`) - checked with `terraform
validate`, but not yet re-verified with a real `terraform plan` (this account's billed resources
were already torn down when it was written) - see `terraform/README.md`.

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
(a cloud destination - S3/Azure Blob/GCS) is a Protocol; `backup.S3BackupTarget` is the one real
implementation (Azure Blob/GCS remain unimplemented extension points), same "S3 instead of a new
database" pattern as `IngestionJobStore`/`S3DocumentStore` - one JSON object per snapshot
(`{prefix}{snapshot_id}.json`). This is the actual fix for local-only backup never being durable
on the live app: an ECS Fargate task's local disk (where `mlops_backups/` lives) doesn't survive
a task restart or redeploy, so a snapshot written only there was gone the moment the container
that wrote it cycled. `BackupManager(target=...)` uploads every snapshot to the target right after
the local write - local file stays a fast working copy, the target is the durable source of
truth; `target=None` (the default for direct construction) keeps local-only behavior unchanged.
`RecoveryManager.restore_from_target(snapshot_id, target, components)` restores from the durable
copy instead of a local path - what a *fresh* task with no local backup history actually needs.
`PlatformManager.restore_backup_from_target(snapshot_id)` wires this through the facade the same
way `restore_backup(path)` already did, raising `ValueError` if its `BackupManager` wasn't built
with a target. `PlatformManager.list_backups()` returns available snapshot ids (from the target
when configured, local files otherwise) so a caller can discover what's restorable without
filesystem/S3-console access. Both are exposed over HTTP as `GET /admin/backups` / `POST
/admin/backups/restore` (see the Wiring section above) - previously neither was reachable outside
a script or the Python REPL, so the automatic scheduled backup had no counterpart a real operator
could actually use. **Wiring**: `service_factory._build_backup_target()` reuses `S3_BUCKET` (same
bucket async ingestion already uses, distinct prefix `MLOPS_BACKUP_S3_PREFIX` default
`mlops_backups/`) rather than provisioning a second bucket - returns `None` (local-only, matching
prior behavior exactly) when `S3_BUCKET` is unset, same opt-in pattern as async ingestion.

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
- Run robustness evaluation: uv run python evaluation/run_robustness_eval.py --dataset evaluation/robustness_dataset.json

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