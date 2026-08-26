# Chapter 5: Generation — LLMs, Answerers, and Fallback

Retrieval (Chapter 4) produces a ranked list of relevant chunks. This chapter covers the final
step: turning those chunks into a written answer. Every implementation here satisfies the same
`Answerer` Protocol (`src/rag/generation/base.py`): one method,
`answer(query, retrieved_chunks) -> str`.

## 1. The shared grounded prompt

Every LLM-backed answerer builds its prompt through one shared function,
`build_grounded_prompt()` (`src/rag/generation/prompt.py`), so every provider is instructed the
same way and cites chunks the same way:

```python
def build_grounded_prompt(query: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"Source {index + 1} ({item.chunk.chunk_id}):\n{item.chunk.text}"
        for index, item in enumerate(retrieved_chunks)
    )
    return (
        "Answer the question using only the provided context. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
```

This is the concrete mechanism behind "grounding" (defined in [Chapter 0](
00-introduction-to-genai-and-rag.md#6-rag-as-a-pipeline-and-where-this-project-fits)): the LLM is
explicitly told to answer *only* from the numbered sources in front of it, and to admit
uncertainty rather than guess when the context doesn't contain the answer. It doesn't force
grounding — an LLM can still ignore the instruction — but it's the standard, effective way to bias
generation toward the retrieved text instead of the model's own memorized (and unverifiable)
beliefs. [Chapter 6](06-guardrails-and-safety.md) covers the safety net that catches it when this
instruction isn't followed.

## 2. `ExtractiveAnswerer` — no LLM at all

`src/rag/generation/extractive_answerer.py`. This is the default generation provider
(`GENERATION_PROVIDER=extractive`) and doesn't call any LLM. Instead, it picks the single sentence
across all retrieved chunks with the highest word-overlap against the query:

```python
for item in retrieved_chunks:
    for sentence in self._sentences(item.chunk.text):
        score = len(query_terms.intersection(self._tokens(sentence)))
        if score > best_score:
            best_score, best_sentence = score, sentence
```

Because it can only ever return text that was literally present in a retrieved chunk, it is
**grounded by construction** — there's no generative step that could introduce a hallucinated
claim, because nothing is generated. The tradeoff: answers are exactly one extracted sentence,
which reads less naturally than something an LLM would write, and it can't synthesize information
spread across multiple chunks the way an LLM summarizing several passages can. It's a genuinely
useful default anyway — deterministic, instant, works with zero cloud credentials, and is what the
test suite and any environment without LLM API access falls back to.

## 3. `BedrockAnswerer` — AWS Bedrock

`src/rag/generation/bedrock_answerer.py`. Takes an already-constructed `boto3` `bedrock-runtime`
client (injected, not built internally — same "caller supplies the authenticated client" pattern
as `OpenSearchVectorStore` in Chapter 3) and a `model_id`.

It calls Bedrock's **Converse API** (`client.converse(...)`) rather than the older, more common
`invoke_model(...)`. This matters for a concrete reason: `invoke_model` requires a
provider-specific JSON request body (Anthropic's shape differs from Amazon Titan's, which differs
from Meta Llama's), so switching model providers means rewriting the request-building code.
Converse normalizes this into one request shape that works across every Bedrock model provider.
It's also required for some newer Claude models (e.g. Claude Haiku 4.5) that Bedrock only exposes
through an **inference profile ARN** rather than a plain model id — Converse accepts `model_id` as
either shape (a plain id like `anthropic.claude-3-haiku-20240307-v1:0`, or a full inference-profile
ARN) without the calling code needing to know or care which.

```python
response = self.client.converse(
    modelId=self.model_id,
    messages=[{"role": "user", "content": [{"text": prompt}]}],
    inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature}
)
```

## 4. `OpenAICompatibleAnswerer` — any Chat Completions endpoint

`src/rag/generation/openai_compatible_answerer.py`. Builds its own `openai.OpenAI` client from
`api_key`/`base_url`/`model_name`, and works against *any* provider that speaks the OpenAI Chat
Completions API shape — not just OpenAI itself. That includes Azure OpenAI, GitHub Models, Ollama
(local models), OpenRouter, and Groq — switching between them is purely a config change
(`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`), never a code change. This project's CI/deployment
setup actually uses **GitHub Models** through this exact adapter — a free-tier LLM API GitHub
provides — which is why "any OpenAI-compatible endpoint" wasn't just a theoretical claim here.

Two behaviors worth calling out specifically:

- **No retrieved chunks → no LLM call.** Returns a fixed fallback string
  (`"I do not know based on the provided context."`) immediately, without spending a request on a
  question that has no context to answer from.
- **Retry only transient failures.** `_is_retryable()` checks the error's HTTP status against
  `{429, 500, 502, 503, 504}` — rate limiting and server-side errors, which are worth retrying
  with exponential backoff (`backoff_base_seconds * 2**attempt`). A `401` (bad API key) or `400`
  (malformed request) is *not* retried, since retrying a request that's wrong by construction just
  wastes time and burns rate-limit budget before failing anyway. After exhausting retries, or on a
  non-retryable failure, it returns a fallback string (`"I couldn't generate an answer at this
  time."`) rather than raising — the caller (`RAGService.ask()`) never has to handle an exception
  from generation.

## 5. `FallbackAnswerer` — provider redundancy

`src/rag/generation/fallback_answerer.py`. The problem it solves: even with retry logic inside
`OpenAICompatibleAnswerer`, a *primary* provider can fail in ways retries can't fix — Bedrock
throttling that outlasts the retry budget, an AWS Marketplace billing/subscription problem (a real
incident hit during this project's own deployment — a specific foundation model requires a
separate Marketplace subscription independent of IAM permissions and account credit balance), or
a network partition. Without a fallback, any of these turns into a 500 error for the end user on
every single question until someone fixes the underlying issue.

`FallbackAnswerer` wraps two `Answerer` instances:

```python
def answer(self, query, retrieved_chunks) -> str:
    try:
        return self.primary.answer(query, retrieved_chunks)
    except Exception as ex:
        logger.warning("primary_answerer_failed_falling_back", ...)
        return self.fallback.answer(query, retrieved_chunks)
```

It's deliberately generic — it implements the `Answerer` Protocol itself and has no knowledge of
which concrete providers it's wrapping, so `RAGService` (or anything else expecting an `Answerer`)
can't tell the difference between talking to a single provider and talking to a
primary-with-fallback pair. Wired in `service_factory.build_rag_service()`:

```python
answerer = _build_answerer(settings.generation_provider, settings, "GENERATION_PROVIDER")
if settings.generation_fallback_provider:
    fallback_answerer = _build_answerer(
        settings.generation_fallback_provider, settings, "GENERATION_FALLBACK_PROVIDER"
    )
    answerer = FallbackAnswerer(primary=answerer, fallback=fallback_answerer)
```

`GENERATION_FALLBACK_PROVIDER` is unset by default — when it's not configured, `RAGService.answerer`
is just the primary provider, completely unchanged from before `FallbackAnswerer` existed. When
set (this project's deployment uses Bedrock as primary and GitHub Models as fallback, or vice
versa depending on which was being validated at the time), any exception from the primary silently
degrades to the secondary rather than failing the request outright.

## 6. Tracing one real question through generation

Continuing the example from earlier chapters — a corpus containing "Contractors receive 10 days
of leave," a user asks *"How many leave days do contractors receive?"*:

1. Retrieval (Chapter 4) returns that chunk as the top (and likely only strongly relevant) result.
2. `build_grounded_prompt()` produces:
   ```
   Answer the question using only the provided context. If the answer is not in the context, say you do not know.

   Context:
   Source 1 (leave_policy.md:0):
   Employees receive 20 days of paid leave annually. Contractors receive 10 days of leave.

   Question: How many leave days do contractors receive?

   Answer:
   ```
3. With `GENERATION_PROVIDER=extractive`: `ExtractiveAnswerer` scores each sentence in that chunk
   against the query terms `{how, many, leave, days, contractors, receive}` — the sentence
   "Contractors receive 10 days of leave." scores highest (5 overlapping terms) and is returned
   verbatim.
4. With an LLM provider instead: the model reads the prompt above and, following the grounding
   instruction, answers something like "Contractors receive 10 days of leave." — phrased in its
   own words but still anchored to that one fact actually present in the context.

This exact scenario is what `tests/unit/test_api.py::test_ingest_and_ask_endpoints` asserts
end-to-end through the real FastAPI `TestClient`.

Next: [Chapter 6 — Guardrails & Safety](06-guardrails-and-safety.md).
