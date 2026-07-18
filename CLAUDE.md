# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Known issue: missing sample document

`sample_documents/AI-RMF-1stdraft.pdf`, referenced by `main.py` and
`tests/unit/test_pdf_parser.py::test_pdf_parser_success`, is not present in the repo (never
committed). `uv run python main.py` and that one test will fail with `FILE_NOT_FOUND` until a
PDF is placed at that path — this is the only remaining gap.

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
   (BAAI/bge-small-en-v1.5) — note `sentence-transformers` is not in `pyproject.toml` dependencies
   yet.
4. **Vector store** (`src/rag/vector_store/`) — `VectorStore` protocol. `InMemoryVectorStore` does
   brute-force cosine similarity over an in-process dict, used for local/dev/tests.
   `OpenSearchVectorStore` is the production adapter; it takes an already-authenticated OpenSearch
   client injected by the caller (never constructs its own client, to keep the core package free
   of AWS/OpenSearch imports).
5. **Retrieval** (`src/rag/retrieval/hybrid_retrieval.py`) — `HybridRetriever` fuses vector cosine
   similarity with a keyword-overlap score (`vector_weight=0.65` / `keyword_weight=0.35`) over an
   over-fetched candidate set (`top_k * 4`), then truncates to `top_k`.
6. **Generation** (`src/rag/generation/`) — `Answerer` protocol. `ExtractiveAnswerer` picks the
   best-overlap sentence from retrieved chunks (no LLM call, fully deterministic — grounded by
   construction since it only ever returns retrieved text). `BedrockAnswerer` is the production
   adapter: takes an injected boto3 `bedrock-runtime` client and builds an Anthropic
   Messages-API-shaped payload for `invoke_model`.

**Wiring:** `app/service_factory.py::build_rag_service()` reads `app/config.py::Settings` (from
env vars: `VECTOR_STORE_PROVIDER`, `EMBEDDING_PROVIDER`, `GENERATION_PROVIDER`, etc.) and
currently only permits the local/default providers (`memory`, `hashing`, `extractive`) —
requesting anything else raises `ServiceConfigurationError` with a message explaining that the
production adapter (`OpenSearchVectorStore`, `BedrockAnswerer`) must be constructed and injected
by the caller directly, since those need authenticated clients the factory doesn't have.

**Evaluation** (`src/evaluation/metrics.py`) is a standalone module of retrieval metrics
(`recall_at_k`, `precision_at_k`, `mean_reciprocal_rank`) — not wired into the service, used for
offline retrieval-quality checks.

Tests in `tests/unit/` mirror this structure one-to-one and mostly test each layer in isolation
via real (non-mocked) local implementations, plus `test_api.py` for an end-to-end
ingest-then-ask flow through the FastAPI `TestClient`.


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