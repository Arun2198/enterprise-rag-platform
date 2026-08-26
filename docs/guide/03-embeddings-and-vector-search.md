# Chapter 3: Embeddings & Vector Search

## 1. What an embedding actually is

An **embedding** is a fixed-length list of numbers (a vector) that represents a piece of text. A
model trained specifically for this task converts text into these numbers such that texts with
similar *meaning* end up as numbers that are close together, no matter how differently they're
worded.

Concretely, this project's default embedder produces a list of 384 or 768 floating-point numbers
per chunk of text (the exact count is called the vector's **dimensionality** — more below). You
can't read meaning off any individual number; the vector only means something as a whole, in
relation to other vectors produced by the same model.

## 2. Why "closeness" between vectors, and how it's measured

Once text is a vector, "is this document relevant to this question" becomes "are these two
vectors close together" — a well-defined math problem instead of a fuzzy language problem. This
project measures closeness with **cosine similarity**: the cosine of the angle between two
vectors, ignoring their length and looking only at the direction they point.

```python
# src/rag/vector_store/in_memory_store.py
def _cosine_similarity(self, first: list[float], second: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(first, second))
    first_norm = sqrt(sum(a * a for a in first))
    second_norm = sqrt(sum(b * b for b in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return numerator / (first_norm * second_norm)
```

The result ranges from `-1` (pointing in opposite directions — unrelated/opposite meaning) to `1`
(pointing in exactly the same direction — same meaning), with `0` meaning no relationship. In
practice, embeddings of genuinely related text usually land somewhere between `0.5` and `0.95`,
rarely close to a perfect `1.0` unless the texts are near-duplicates.

**A worked example**, with toy 2-dimensional vectors for intuition (real vectors have hundreds of
dimensions, but the math is identical):

- Query "vacation days" embeds to `[0.9, 0.1]`.
- Chunk A, "employees receive 20 days of paid leave," embeds to `[0.85, 0.2]` (close direction —
  similar meaning).
- Chunk B, "the office is located in downtown," embeds to `[0.1, 0.95]` (very different direction
  — unrelated meaning).

Cosine similarity of query vs. Chunk A: high (both vectors point mostly along the first axis).
Cosine similarity of query vs. Chunk B: low (the vectors point in almost perpendicular
directions). Chunk A ranks above Chunk B — exactly the outcome you want, even though "vacation
days" and "paid leave" share no common words at all. This is the entire point of embeddings over
plain keyword search: they capture paraphrase, not just exact wording.

## 3. This project's two embedders

Both implement the same `Embedder` Protocol (`src/rag/embeddings/base.py`): one method,
`embed(text: str) -> list[float]`.

### `HashingEmbedder` — deterministic, offline, zero model download

`src/rag/embeddings/hashing_embedder.py`. It does **not** understand meaning at all — it produces
a vector purely from which words are present, using a hash function:

```python
def embed(self, text: str) -> list[float]:
    vector = [0.0] * self.dimensions   # default 384
    for token in self._tokens(text):                     # lowercase word tokens
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % self.dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    ...                                                    # then L2-normalize
```

Each distinct word deterministically hashes to one of 384 vector positions and nudges it up or
down. Two texts sharing many of the same words end up with similar vectors; two texts using
different words for the same idea (like "vacation days" vs. "paid leave") do **not** — this
embedder has no concept of synonyms or paraphrase, only literal word overlap.

Why it exists at all: it needs no model download, no GPU/CPU-heavy inference, and no network
call, so it makes the test suite ([Chapter 12](12-testing-strategy.md)) and offline/quick-script
use fast and fully deterministic. `RAGService()` (direct construction, used by tests and scripts)
defaults to this embedder for exactly that reason. **It is never the default for the live,
deployed app** — see section 5.

### `SentenceTransformerEmbedder` — a real neural embedding model

`src/rag/embeddings/sentence_transformer_embedder.py`, wrapping the open-source
[`sentence-transformers`](https://www.sbert.net/) library, loading a real pretrained neural
network:

```python
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)   # downloads/loads the model once
        self.dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
```

The model is loaded once, at construction, and reused for every subsequent call — constructing a
fresh model per request would be far too slow. This actually understands semantic similarity,
because it was trained on huge amounts of text specifically to place similar meanings close
together in vector space — it's what makes the "vacation days" / "paid leave" example above work
for real, not just as a toy illustration.

**The model in use**: this project's configured default (via `EMBEDDING_MODEL_NAME` in
`src/app/config.py`) is `BAAI/bge-base-en-v1.5` — not the class's own fallback default of
`BAAI/bge-small-en-v1.5` shown above (that fallback only matters if the class were constructed
with no arguments at all, which the live app never does — `service_factory.py` always passes
`model_name=settings.embedding_model_name` explicitly). BGE ("BAAI General Embedding") is a
well-regarded, fully open-source (MIT-licensed) embedding model family with no per-call API cost
and no data leaving the machine it runs on — this was a deliberate choice over embedding APIs like
OpenAI's, driven by an explicit project directive to never default to a "substandard" embedding
for the deployed app.

## 4. Vector storage and search

Both stores implement the `VectorStore` Protocol (`src/rag/vector_store/base.py`): `add()`,
`add_many()`, `search(query_embedding, top_k, metadata_filter=None) -> list[SearchResult]`.

### `InMemoryVectorStore` — brute-force, in-process

`src/rag/vector_store/in_memory_store.py`. Literally a Python dict keyed by `chunk_id`, mapping to
`(Chunk, embedding)` pairs. `search()` computes cosine similarity between the query vector and
*every single stored vector*, sorts descending, and returns the top `top_k`. This is
**O(n)** — it checks every chunk, every time — which is completely fine at the scale this project
targets (a demo corpus, or "hundreds to low thousands" of chunks) and is trivial to reason about
and test, but would become a real bottleneck at genuinely large scale (millions of chunks), where
a proper approximate-nearest-neighbor index becomes necessary. It also supports an optional
`metadata_filter` (exact-match on chunk metadata fields, e.g. restricting search to one
`document_type`), applied before scoring.

### `OpenSearchVectorStore` — the production adapter

`src/rag/vector_store/opensearch_store.py`. Talks to AWS OpenSearch's k-NN (k-nearest-neighbor)
search API instead of scanning in Python. It takes an **already-authenticated** OpenSearch client
as a constructor argument — it never builds its own client or imports AWS-specific auth code — so
the core `rag` package stays free of any AWS/OpenSearch dependency unless you actually choose to
use this adapter. This is the same provider-swap pattern from [Chapter 1](
01-project-overview.md#4-the-core-architectural-idea-provider-swap-via-protocols): identical
`VectorStore` interface, different backend, zero changes anywhere else in the pipeline. As of this
writing, this adapter is available but must be constructed and injected manually — the live
deployment (Chapter 10) still runs `InMemoryVectorStore`, since `service_factory.py` currently
only wires `VECTOR_STORE_PROVIDER=memory` and raises `ServiceConfigurationError` for anything
else, specifically because OpenSearch needs an externally-managed authenticated client the factory
doesn't build for you.

## 5. Where this is wired for the live app

`service_factory._build_embedder()` (`src/app/service_factory.py`) is the single place that
decides which embedder the deployed app actually uses:

```python
def _build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider not in WIRED_EMBEDDING_PROVIDERS:   # ("hashing", "sentence_transformer")
        raise ServiceConfigurationError(...)
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name=settings.embedding_model_name)
```

`EMBEDDING_PROVIDER` defaults to `sentence_transformer` (`src/app/config.py`) — so the live app
gets the real neural model unless someone explicitly opts into `hashing`. This same embedder
instance is also handed to the `HallucinationDetector` guardrail (`_build_guardrail_manager`,
[Chapter 6](06-guardrails-and-safety.md)) rather than each independently constructing its own —
keeping groundedness scoring consistent with whatever embedding space retrieval is actually
using.

## 6. Keeping tests fast without ever using a "fake" quality model

Downloading and running a real neural network in every one of the 400+ unit tests would be slow
and would need network access on every test run. Instead, `tests/unit/conftest.py` globally
replaces the `SentenceTransformer` class itself, at test-session start, with a fake:

```python
patch(
    "rag.embeddings.sentence_transformer_embedder.SentenceTransformer",
    _FakeSentenceTransformer
).start()
```

`_FakeSentenceTransformer` internally delegates to `HashingEmbedder` — so any test that goes
through `service_factory.build_rag_service()` (including `app.main`'s own module-level
construction) never downloads a real model or makes a network call, while still getting
deterministic, content-sensitive vectors that behave sensibly for ranking assertions ("chunk A
should outscore chunk B for this query"). This patch is started once and never stopped — it's a
permanent substitution for the whole test process, not a per-test mock.

Next: [Chapter 4 — Retrieval & Reranking](04-retrieval-and-reranking.md).
