# Chapter 13: Security, Known Gaps, and Glossary

This closing chapter has two purposes: an honest, factual accounting of what this project
deliberately does and doesn't handle from a security standpoint, and a glossary tying together
every term used across this guide.

## 1. A real security fix, walked through in detail: the `/ingest` path traversal

This is worth covering as a concrete case study, not just a bullet point, because it's a genuine
example of how an innocent-looking API can become a serious vulnerability.

**The vulnerability**: `/ingest` accepts a JSON body of file paths (`{"file_paths": [...]}`) and,
before the fix, handed each one straight to the ingestion pipeline with **zero validation**. This
is an unauthenticated HTTP endpoint ([section 2](#2-what-authentication-this-project-does-and-does-not-have)
below covers why that matters even more). A request like:

```json
{"file_paths": ["../../etc/passwd"]}
```

or, on the deployed container, any absolute path reachable inside it —
`{"file_paths": ["/app/src/app/config.py"]}` or similar — would get parsed, chunked, embedded, and
indexed by whatever parser matched its extension. Once indexed, its content becomes retrievable
through `/ask` — a completely unauthenticated caller could read arbitrary files off the container's
filesystem, one `/ingest` + `/ask` round trip at a time. `../../` **path traversal** (using `..`
segments to walk outside an intended directory) combined with an endpoint that takes attacker-
controlled paths and reads whatever they point to is a well-known, serious vulnerability class —
this wasn't a theoretical or contrived scenario, it was a real gap in a real endpoint.

**The fix** (`RAGService._is_path_allowed()`, [Chapter 2](
02-ingestion-and-chunking.md#5-tracing-a-real-ingest-call-end-to-end)):

```python
def _is_path_allowed(self, file_path: str) -> bool:
    if self.ingest_allowed_dir is None:
        return True
    try:
        resolved = Path(file_path).resolve()
    except (OSError, ValueError):
        return False
    return resolved == self.ingest_allowed_dir or self.ingest_allowed_dir in resolved.parents
```

Two things matter here: `Path.resolve()` normalizes the path — collapsing `..` segments and
following symlinks — *before* any comparison happens, so a traversal attempt can't sneak past a
naive string-prefix check (`"../../etc/passwd".startswith("/app/sample_documents")` is `False`
even though the string itself doesn't look obviously wrong at a glance; `resolve()` turns it into
its real, final destination first). Then the check requires the resolved path to actually sit
*inside* the allowed directory — either be that directory itself, or have it as one of its
parents — rejecting anything outside with a `PATH_NOT_ALLOWED` error rather than reading it.

`ingest_allowed_dir` defaults to `None` (unrestricted) on **direct** `RAGService()` construction —
this is intentional, not an oversight: tests and trusted local scripts construct `RAGService`
directly and need to read arbitrary local test fixtures. `service_factory.build_rag_service()` —
what the live, network-facing app actually uses — **always** sets it, from `INGEST_ALLOWED_DIR`
(default `sample_documents`). The permissive default exists for trusted callers; the live app is
never one of them.

## 2. What authentication this project does and does not have

**There is no authentication on any endpoint.** `/ingest`, `/ask`, and the `/admin/*` endpoints
are all reachable by anyone who can reach the deployed URL, with no API key, login, or token
required. This is the single most consequential gap for anything beyond a demo — it means the
path-traversal fix in section 1 matters *because* there's nothing else standing between an
arbitrary internet caller and the ingestion pipeline. It also means the `/admin/feature-flags` and
`/admin/scheduler/jobs` endpoints ([Chapter 8](08-mlops-platform.md)) — which can change live
system behavior — are equally unauthenticated.

This is a known, acknowledged gap, not a hidden one. `src/mlops/permissions.py`'s RBAC system
([Chapter 8](08-mlops-platform.md#6-the-rest-of-the-platform-built-tested-not-yet-wired-into-the-request-path))
exists as a building block for *authorization* (once you know who's calling, what are they allowed
to do) but there's nothing upstream of it that establishes *authentication* (who is actually
calling) — no login flow, no session, no token validation anywhere in this codebase.

## 3. AWS IAM: not least-privilege

Every IAM role this deployment uses (`ecsTaskExecutionRole`, `ecsTaskRole`,
`ecsInfrastructureRoleForExpressServices`, `github-actions-ecs-deploy-role`) currently carries
broad, AWS-managed `*FullAccess` policies rather than custom policies scoped to exactly the
specific actions each role actually needs. This was a deliberate, explicitly acknowledged tradeoff
for demo speed during a real, iterative deployment build-out ([Chapter 10](10-aws-deployment.md))
— narrowing each role down to least-privilege was flagged repeatedly as future work and never
done. The practical risk: if any of these roles' credentials were ever misused (e.g. through a
vulnerability in the running application itself), the blast radius is far larger than it needs to
be — a `*FullAccess` policy can do far more than "run this one container and call Bedrock."

## 4. Everything else, listed honestly

Compiled directly from this project's own audit findings (not softened or omitted):

| Gap | Why it matters |
|---|---|
| No authentication anywhere | Covered in section 2 — the most consequential single gap |
| IAM not least-privilege | Covered in section 3 |
| No private VPC networking | The ALB and ECS tasks sit in default networking rather than an isolated private subnet setup |
| No rate limiting | An unauthenticated caller could send unlimited `/ask` requests, each of which costs real LLM API/Bedrock spend |
| No CI test gate before deploy | Covered in [Chapter 11](11-cicd-and-github-actions.md) — `deploy-aws.yml` doesn't run `pytest` before deploying |
| ECR image scanning disabled | Vulnerabilities in base image/dependencies wouldn't be automatically flagged |
| No CloudWatch alarms/dashboards | No automated alerting if the deployed service degrades or errors spike |
| No SLIs/SLOs defined | No formally agreed target for latency/availability to alert against in the first place |
| No query/access audit logging | No record of who asked what, beyond whatever the application's own `logging` calls happen to capture |
| Inconsistent dependency version pinning in `pyproject.toml` | Some dependencies pinned tightly, others loosely — not a uniform policy |
| `pytest`/`pytest-cov` ship in the production image | Test tooling isn't separated into a dev-only dependency group, so the deployed container is slightly larger than it needs to be |
| Golden dataset ids are positional, not content-addressed | Covered in [Chapter 7](07-evaluation-framework.md) — a permanent characteristic of the current evaluation design, not a bug, but easy to forget when changing chunking |
| Drift detection, retraining workflows | Protocol-only, unimplemented — [Chapter 8](08-mlops-platform.md#6-the-rest-of-the-platform-built-tested-not-yet-wired-into-the-request-path) |
| Toxicity/hate-speech classification, BERTScore, RAGAS | Deliberately deferred — [Chapter 6](06-guardrails-and-safety.md#4-optional-guardrails--same-interface-not-on-by-default), [Chapter 7](07-evaluation-framework.md#7-layer-2--generation-quality) |
| `OpenSearchVectorStore` not wired for the live app | Built and tested, but requires a caller-supplied authenticated client the factory doesn't build — [Chapter 3](03-embeddings-and-vector-search.md#4-vector-storage-and-search) |

None of this is presented to alarm — a demo/portfolio-scale RAG platform reasonably defers
production-hardening work like this. The point of listing it here explicitly is that this guide
promised factual completeness, including about what isn't done, not just what is.

## 5. Glossary

**Answerer** — this project's term for the component that turns retrieved chunks into a written
answer; a `Protocol` with extractive, Bedrock, OpenAI-compatible, and fallback implementations.
[Ch 5](05-generation-and-llms.md)

**Chunk** — a small, search-sized piece of a document, the unit both indexing and retrieval
operate on. [Ch 2](02-ingestion-and-chunking.md)

**Chunking** — splitting a document's text into chunks. [Ch 2](02-ingestion-and-chunking.md)

**Cosine similarity** — a measure of how similar two vectors' directions are, ignoring magnitude;
this project's method for comparing embeddings. [Ch 3](03-embeddings-and-vector-search.md)

**Container / Docker image** — a self-contained, portable package of an application plus
everything it needs to run. [Ch 9](09-containers-and-docker.md)

**Cross-encoder** — a model that scores a query and a candidate together, as one joint input,
rather than independently — more accurate than a bi-encoder, more expensive to run at scale.
[Ch 4](04-retrieval-and-reranking.md)

**Embedding** — a fixed-length numeric vector representing a piece of text's meaning, positioned
so that similar meanings land close together. [Ch 0](00-introduction-to-genai-and-rag.md),
[Ch 3](03-embeddings-and-vector-search.md)

**Fine-tuning** — retraining a model's own parameters on new data (not used anywhere in this
project — RAG is the alternative approach taken instead). [Ch 0](00-introduction-to-genai-and-rag.md)

**Grounding / grounded answer** — an answer whose claims trace back to retrieved source text,
rather than a model's own unverifiable memory. [Ch 0](00-introduction-to-genai-and-rag.md),
[Ch 5](05-generation-and-llms.md)

**Guardrail** — an automated safety check run before (input stage) or after (output stage)
generation — PII detection, hallucination detection, etc. [Ch 6](06-guardrails-and-safety.md)

**Hallucination** — an LLM confidently generating plausible but false or unsupported content.
[Ch 0](00-introduction-to-genai-and-rag.md)

**Hybrid retrieval** — combining vector (semantic) search with keyword overlap scoring.
[Ch 4](04-retrieval-and-reranking.md)

**IAM (Identity and Access Management)** — AWS's permission system; nothing is allowed by default.
[Ch 10](10-aws-deployment.md)

**LLM (Large Language Model)** — a model trained to predict likely next tokens given prior text,
producing fluent generated language. [Ch 0](00-introduction-to-genai-and-rag.md)

**OIDC federation** — using short-lived, cryptographically verifiable identity tokens (here, from
GitHub Actions) instead of a stored long-lived credential, to authenticate to AWS.
[Ch 10](10-aws-deployment.md)

**Path traversal** — an attack using `..`/absolute-path segments to escape an intended directory
restriction; the vulnerability class behind the `/ingest` fix in section 1.

**Protocol** — Python's structural-typing interface mechanism; the basis of this project's
provider-swap architecture. [Ch 1](01-project-overview.md)

**RAG (Retrieval-Augmented Generation)** — answering questions by retrieving relevant source text
first, then generating an answer grounded in it, rather than relying on a model's own memorized
knowledge. [Ch 0](00-introduction-to-genai-and-rag.md)

**Reranking** — a second, more careful relevance-scoring pass over an initial retrieval candidate
set, typically using a cross-encoder. [Ch 4](04-retrieval-and-reranking.md)

**`Result[T]`** — this project's typed success/data/error return-value pattern, used instead of
exceptions for expected failure modes in ingestion. [Ch 2](02-ingestion-and-chunking.md)

**Token** — the unit an LLM actually processes text in — often a word or word-fragment.
[Ch 0](00-introduction-to-genai-and-rag.md)

**Vector store** — where chunk embeddings are stored and searched; `InMemoryVectorStore` (brute-
force, local) or `OpenSearchVectorStore` (AWS-managed) in this project.
[Ch 3](03-embeddings-and-vector-search.md)

---

That's the end of the guide. If something here goes stale — a default changes, a gap gets closed,
a new provider gets wired in — the chapter describing it should be updated to match, the same way
this guide was written: by reading the actual current code, not by editing prose in isolation.
