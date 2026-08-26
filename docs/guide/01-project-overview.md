# Chapter 1: Project Overview & How to Run It

## 1. What this project is

`enterprise-rag-platform` is a working RAG service: you give it document files, it indexes them,
and then you can ask it natural-language questions and get answers grounded in those documents,
with source citations. It's built as a FastAPI web service with three HTTP endpoints
(`/ingest`, `/ask`, `/health`), backed by a layered pipeline, plus a standalone evaluation
framework and an operational ("MLOps") backbone for running it as a real, monitored service rather
than a one-off script.

It deploys to AWS (ECS Fargate) via GitHub Actions, but every piece also runs entirely locally with
no cloud account and no API keys required — that split (local-first defaults, cloud as an
explicit opt-in) is the central design decision of the whole codebase, explained in section 4
below.

## 2. Tech stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.12 | |
| Dependency management | [`uv`](https://docs.astral.sh/uv/) | fast, lockfile-based (`uv.lock`), replaces pip+venv |
| Web framework | FastAPI | async, automatic request/response validation via Pydantic, built-in OpenAPI docs |
| Data validation | Pydantic v2 | schemas in `app/schemas.py` define every API request/response shape |
| PDF/DOCX parsing | `pypdf`, `python-docx` | |
| Embeddings | `sentence-transformers` (model: `BAAI/bge-base-en-v1.5`) | local, open-source, no API cost — see [Ch 3](03-embeddings-and-vector-search.md) |
| Reranking | `sentence-transformers` `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) | see [Ch 4](04-retrieval-and-reranking.md) |
| LLM generation | AWS Bedrock (`boto3`) or any OpenAI-compatible endpoint (`openai` client) | see [Ch 5](05-generation-and-llms.md) |
| PII detection (optional) | Microsoft Presidio + spaCy `en_core_web_sm` | see [Ch 6](06-guardrails-and-safety.md) |
| Observability | OpenTelemetry API/SDK | metrics hooks, no exporter wired by default |
| Testing | `pytest`, `pytest-cov` | 400+ tests, all offline (see [Ch 12](12-testing-strategy.md)) |
| Containerization | Docker | see [Ch 9](09-containers-and-docker.md) |
| Cloud | AWS (ECS Fargate, ECR, Bedrock, IAM/OIDC) | see [Ch 10](10-aws-deployment.md) |
| CI/CD | GitHub Actions | see [Ch 11](11-cicd-and-github-actions.md) |

Full dependency list: [`pyproject.toml`](../../pyproject.toml).

## 3. The system in one picture

```
                         ┌─────────────────────────────┐
   POST /ingest ────────►│                              │
   (file paths)          │        RAGService            │
                         │  (app/services/rag_service.py)│
   POST /ask ───────────►│                              │──► AskResponse
   (a question)          └───────────────┬──────────────┘    (answer + sources
                                          │                    + guardrail_flags)
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
       IngestionPipeline           HybridRetriever              GuardrailManager
       (parse + chunk)           + CrossEncoderReranker         (input + output
              │                           │                      safety checks)
              ▼                           ▼                           ▲
        Embedder + VectorStore      Answerer (LLM or                  │
        (index chunks)               extractive) ────────────────────┘
```

Every box in this diagram is a `Protocol` (Python's structural interface) with at least one real,
runnable implementation and no hidden cloud dependency in the default path. That's the "provider-
swap" pattern mentioned throughout this guide — explained concretely in section 4.

## 4. The core architectural idea: provider-swap via Protocols

This is the single most important thing to understand about how this codebase is organized, so
it's worth explaining from first principles.

A **`Protocol`** in Python (from the `typing` module) defines a shape — a set of methods an object
must have — without requiring that object to inherit from any particular base class. It's Python's
version of an interface. For example, every embedding provider in this project satisfies:

```python
# src/rag/embeddings/base.py (paraphrased)
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Anything with an `embed()` method matching that signature *is* an `Embedder` as far as the rest of
the code is concerned — no inheritance needed. This project defines one of these Protocols for
every pipeline stage (parsing, embedding, vector storage, retrieval, generation, guardrails), and
then provides:

- **A local, dependency-free (or cheap) default implementation** that runs on a laptop with no
  cloud account, no API key, and (for most stages) no large model download — e.g.
  `HashingEmbedder`, `InMemoryVectorStore`, `ExtractiveAnswerer`.
- **One or more production adapters** that wrap a real external system — e.g.
  `SentenceTransformerEmbedder` (a real local ML model), `OpenSearchVectorStore` (AWS managed
  search), `BedrockAnswerer` (AWS's LLM API).

Swapping which one is active is never a code change — it's just passing a different object into
`RAGService.__init__()`. Nothing is wired through a dependency-injection framework or a config
file the pipeline code reads directly; `RAGService` just holds references to whatever objects it
was constructed with and calls their Protocol methods.

**Two different callers construct `RAGService` differently, on purpose:**

- **Direct construction** (`RAGService()` with no arguments) — used by the test suite and by
  quick scripts. It gets fast, offline, zero-dependency defaults: `HashingEmbedder`,
  `InMemoryVectorStore`, `ExtractiveAnswerer`, no guardrails, no reranker, no ingest path
  restriction. This exists so the test suite runs in seconds with no network calls (see
  [Ch 12](12-testing-strategy.md)).
- **`service_factory.build_rag_service(settings)`** — used by the live app (`app/main.py`) and by
  `main.py`'s demo script. It reads environment variables (via `Settings`, shown in section 5
  below) and wires in the real, production-quality defaults: the real sentence-transformer
  embedding model, the cross-encoder reranker, the PII/hallucination guardrails, and a path
  restriction on `/ingest`.

This split matters: if you construct `RAGService()` directly in your own script, you get the
*fast, offline* defaults, not the *high-quality* ones. `main.py`'s demo script deliberately goes
through `build_rag_service()` for exactly this reason — it's meant to show what the deployed app
actually does, not the offline fallback.

`service_factory.py` is also a deliberate **gate**: if you set an environment variable to a
provider name the factory doesn't recognize (say, `EMBEDDING_PROVIDER=some_typo`), it raises a
`ServiceConfigurationError` immediately at startup rather than silently falling back to something
else. This is a conscious tradeoff — a loud startup crash for a typo'd config value is much easier
to debug than a service that's silently running with the wrong provider.

## 5. Configuration

Every tunable behavior is an environment variable, read once at startup into a single immutable
`Settings` dataclass (`src/app/config.py`). Some of the most load-bearing ones:

| Variable | Default | What it controls |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence_transformer` | `sentence_transformer` (real model) or `hashing` (offline, deterministic — see [Ch 3](03-embeddings-and-vector-search.md)) |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | which sentence-transformers model to load |
| `GENERATION_PROVIDER` | `extractive` | `extractive` (no LLM), `openai_compatible`, or `bedrock` — see [Ch 5](05-generation-and-llms.md) |
| `GENERATION_FALLBACK_PROVIDER` | unset | a second provider to fall back to if the primary throws |
| `RERANKER_ENABLED` | `true` | whether the cross-encoder reranking stage runs |
| `GUARDRAILS_ENABLED` | `true` | master switch for the whole guardrails stage |
| `INGEST_ALLOWED_DIR` | `sample_documents` | directory `/ingest` is restricted to (security control, see [Ch 13](13-security-and-glossary.md)) |
| `MLOPS_ENABLED` | `true` | whether feature flags, the scheduler, and admin endpoints are active |

Every variable, its default, and its parsing lives in [`src/app/config.py`](../../src/app/config.py)
— that file is the single source of truth, not this table.

## 6. Running it yourself

Everything below assumes `uv` is installed and you're in the repo root.

Install dependencies (reads `uv.lock`, creates a local virtual environment):

```bash
uv sync
```

Run the demo script — ingests the sample NIST AI Risk Management Framework PDF
(`sample_documents/AI-RMF-1stdraft.pdf`) and asks it a question end to end, printing the answer
and its source chunks:

```bash
uv run python main.py
```

Run the API locally (auto-reloads on code changes):

```bash
uv run uvicorn app.main:app --reload --app-dir src
```

With the API running, in another terminal:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["sample_documents/AI-RMF-1stdraft.pdf"]}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the goal of the AI RMF?", "top_k": 4}'
```

Run the full test suite (400+ tests, all offline, no model downloads — typically finishes in
under a minute):

```bash
uv run pytest
```

Run one test file, or one specific test, while iterating:

```bash
uv run pytest tests/unit/test_rag_service.py
uv run pytest tests/unit/test_rag_service.py::test_name -v
```

Run the retrieval-quality evaluation suite against the golden dataset (see
[Ch 7](07-evaluation-framework.md) for what this actually measures):

```bash
uv run python evaluation/run_eval.py --dataset evaluation/golden_dataset.json --k 1 3 5 --json
```

## 7. Project layout

```
src/
├── app/         FastAPI routes (main.py), config.py, service_factory.py wiring,
│                schemas.py (Pydantic request/response contracts)
├── ingestion/   parsers (pdf/docx/markdown), Result[T] contracts, no raised exceptions
├── rag/         chunking, embeddings, vector store, retrieval, reranking, generation, guardrails
├── evaluation/  standalone retrieval/generation quality framework
└── mlops/       registries, lifecycle, feature flags, scheduler, governance, RBAC

tests/unit/          mirrors src/ one-to-one, one test file per module
evaluation/           golden dataset (golden_dataset.json) + CLI entry point (run_eval.py)
sample_documents/     the real PDF backing the demo, golden dataset, and tests
.github/workflows/    CI + AWS deployment pipeline
docs/guide/            this guide
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, which is why test files import as
`from app...`, `from rag...`, `from ingestion...`, `from evaluation...` — not `from src...`.

Next: [Chapter 2 — Document Ingestion & Chunking](02-ingestion-and-chunking.md).
