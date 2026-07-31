# Enterprise RAG Platform

A Retrieval-Augmented Generation service for querying enterprise documents in natural language,
grounded in real retrieved content rather than model recall. Built as a layered pipeline with a
provider-swap architecture: every stage (parsing, embedding, vector storage, generation) is a
`Protocol` with a local, dependency-free default plus injectable production adapters, so nothing
requires cloud credentials to run and test, but everything can be pointed at real infrastructure
without touching the pipeline logic itself.

## What's here

- **Core RAG pipeline** — PDF/DOCX/Markdown ingestion, heading- and sentence-aware chunking,
  hybrid (vector + keyword) retrieval, cross-encoder reranking, and pluggable generation
  (extractive baseline, AWS Bedrock, or any OpenAI-compatible endpoint, with automatic fallback
  between providers).
- **Guardrails** — PII detection/redaction (regex or Presidio NER), hallucination detection
  (token-overlap, NLI, or LLM-as-judge), prompt injection and secret-leakage checks, all
  implementing one `Guardrail` interface with a severity/action taxonomy and audit trail.
- **Evaluation framework** — golden-dataset-driven retrieval metrics (Recall@K, MRR, NDCG,
  Precision@K), generation-quality scoring, system metrics, and run-over-run experiment tracking.
- **MLOps platform** — model/artifact registries, a lifecycle promotion state machine, feature
  flags with canary rollout, a scheduler, governance/audit logging, and backup/recovery — all
  usable standalone or composed through one `PlatformManager` facade.
- **AWS deployment** — containerized, deployed to ECS Fargate (Express Mode) via GitHub Actions
  with OIDC (no long-lived AWS keys), fronted by an ALB with an auto-provisioned TLS certificate.

See [CLAUDE.md](CLAUDE.md) for the full architectural writeup — request flow, every module's
responsibility, and the reasoning behind each design decision.

## Getting started

Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run the demo script — ingests the sample NIST AI RMF PDF and asks it a question end to end:

```bash
uv run python main.py
```

Run the API locally:

```bash
uv run uvicorn app.main:app --reload --app-dir src
```

Then:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_paths": ["sample_documents/AI-RMF-1stdraft.pdf"]}'

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the goal of the AI RMF?", "top_k": 4}'
```

## Testing

```bash
uv run pytest                                    # full suite
uv run pytest tests/unit/test_rag_service.py -v  # a single file
```

No unit test downloads a real model or makes a network call — heavy dependencies
(sentence-transformers, cross-encoders) are faked at the test-session level.

## Evaluation

```bash
uv run python evaluation/run_eval.py --dataset evaluation/golden_dataset.json --k 1 3 5 --json
```

Add `--reranker` to enable cross-encoder reranking, `--generation extractive` (or
`openai_compatible`) to also score generation quality, `--system-metrics` for throughput/cost
estimates, and `--track --trend 10` to record and compare against run history. See CLAUDE.md's
Evaluation section for the full flag list.

## Configuration

Every provider is selected via environment variable and gated by `app/service_factory.py`, which
raises a clear `ServiceConfigurationError` for anything not explicitly wired rather than failing
silently. Key variables:

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence_transformer` | or `hashing` (deterministic, offline, no model download) |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-base-en-v1.5` | |
| `GENERATION_PROVIDER` | `extractive` | or `openai_compatible`, `bedrock` |
| `GENERATION_FALLBACK_PROVIDER` | unset | wraps the primary provider in `FallbackAnswerer` |
| `RERANKER_ENABLED` | `true` | |
| `GUARDRAILS_ENABLED` | `true` | master switch; individual guards have their own flags |
| `INGEST_ALLOWED_DIR` | `sample_documents` | `/ingest` rejects any path outside this directory |
| `MLOPS_ENABLED` | `true` | feature flags, scheduler, admin endpoints |

Full list in `src/app/config.py`.

## Deployment

The app is containerized (`Dockerfile`) and deploys to AWS ECS Fargate via
`.github/workflows/deploy-aws.yml` on every push to `main`. It authenticates to AWS through a
GitHub OIDC trust role rather than stored credentials, builds and pushes to ECR, then deploys via
ECS Express Mode, which provisions the load balancer, target groups, and TLS certificate
automatically. See CLAUDE.md's Wiring section for the full one-time AWS setup this depends on.

## Project layout

```
src/
├── app/         FastAPI routes, config, service_factory wiring
├── ingestion/   parsers (pdf/docx/markdown), Result[T] contracts, no raised exceptions
├── rag/         chunking, embeddings, vector store, retrieval, reranking, generation, guardrails
├── evaluation/  standalone retrieval/generation quality framework
└── mlops/       registries, lifecycle, feature flags, scheduler, governance, RBAC

tests/unit/      mirrors src/ one-to-one
evaluation/      golden dataset + CLI entry point
sample_documents/  real PDF backing the demo, golden dataset, and tests
```
