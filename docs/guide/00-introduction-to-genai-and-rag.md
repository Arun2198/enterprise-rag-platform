# Chapter 0: Introduction to GenAI, LLMs, and RAG

This chapter has no code in it. It exists so that every later chapter can use words like
"embedding," "token," or "hallucination" without stopping to define them. If you already know
what an LLM is and why RAG exists, skip to [Chapter 1](01-project-overview.md).

## 1. What "generative AI" actually means

Most software is deterministic: you give it an input, it runs a fixed set of rules, and it
produces an output. A calculator doesn't "decide" that 2+2=4; it computes it.

A **generative AI model** is different. It's a statistical model trained on enormous amounts of
text (or images, audio, etc.) to predict what's likely to come next, given what came before. When
you type "The capital of France is" into a language model, it doesn't look up an answer in a
database — it calculates, based on patterns learned from its training data, that the word "Paris"
is overwhelmingly the most probable next word, and outputs it.

Do this one word (technically, one **token** — see below) at a time, feeding each output back in
as input for the next prediction, and you get fluent, coherent paragraphs. That's a **Large
Language Model (LLM)**: "large" because it has billions of internal parameters, tuned on massive
text datasets, that encode these next-token probabilities.

## 2. Tokens

Models don't read text letter by letter or word by word. They break text into **tokens** — chunks
that are often a word, sometimes part of a word, sometimes a single punctuation mark. "Retrieval-
Augmented" might become tokens like `Retrieval`, `-`, `Aug`, `mented`. This matters practically in
this project because:

- LLM APIs charge and rate-limit by token count, not character count.
- `evaluation/generation_metrics.py`'s cost estimate and `system_metrics.py`'s "estimated
  completion tokens" (see [Chapter 7](07-evaluation-framework.md)) are token-based approximations.
- Every LLM has a maximum context length measured in tokens — a hard ceiling on how much text
  (prompt + retrieved chunks + conversation history) can fit into a single request.

## 3. What an LLM can't do reliably: knowledge and hallucination

An LLM's "knowledge" is frozen at whatever point its training data was collected, and it exists
only as statistical patterns baked into its parameters — not as a lookup table it can cite or
verify. Two consequences matter a lot for this project:

**It doesn't know about your private data.** A public LLM (GPT, Claude, Llama, whatever) was never
trained on your company's internal PDFs, your product's documentation, or last week's meeting
notes. Ask it a question about them and it has nothing to draw on.

**It can "hallucinate."** Because an LLM is fundamentally a next-token predictor, not a fact
database, it will confidently generate plausible-sounding text even when it has no real basis for
it. Ask an LLM a question it doesn't actually know the answer to, and instead of saying "I don't
know," it will often produce a fluent, convincing, entirely fabricated answer — the model isn't
lying (it has no concept of truth to lie about), it's doing exactly what it was trained to do:
predict likely-sounding text. This is the single biggest reliability problem in applying LLMs to
real work, and it's the reason this whole project exists.

## 4. The two bad options, and the better one

Faced with "the LLM doesn't know my private data and might hallucinate," there are two obvious but
flawed fixes:

**Option A: Fine-tune the model on your data.** Expensive, slow, has to be redone every time your
data changes, and doesn't actually solve hallucination — a fine-tuned model can still fabricate
things confidently, just about your data instead of the world's.

**Option B: Paste everything into the prompt.** Just include your entire document set in every
request. This breaks immediately at any real scale — a company's documentation is far larger than
any model's context window, and even where it would fit, you'd be paying (in tokens and latency)
to re-send the same documents on every single question.

**Option C — Retrieval-Augmented Generation (RAG) — is what this project implements.** Instead of
the model trying to "know" your data, or you trying to force-feed it everything, you:

1. Take the user's question.
2. Search your own document collection for the small number of passages actually relevant to that
   question (this is "retrieval").
3. Hand the LLM only those passages, plus the question, and instruct it to answer *using only
   what's in front of it*.
4. The LLM's job shrinks from "recall facts from training" to "read this short excerpt and
   summarize/answer from it" — something LLMs are reliably good at.

This is why RAG reduces (never fully eliminates) hallucination: the model is reasoning over text
that's actually there, not reaching into its own statistical memory. It also means the system's
"knowledge" is exactly as current as your document collection — update the documents, and the next
question gets a different, up-to-date answer, with no retraining involved.

## 5. Embeddings and semantic search, in one paragraph

Step 2 above ("search for relevant passages") could be done with simple keyword matching, but that
misses paraphrases — a document that says "staff are entitled to annual leave" won't keyword-match
a query asking about "vacation days." The fix is to convert both the query and every document
passage into a list of numbers (a **vector**, more specifically an **embedding**) using a model
trained so that passages with *similar meaning* end up as *nearby* vectors, regardless of exact
wording. Finding relevant passages then becomes a math problem: find the stored vectors closest to
the query's vector. Chapter 3 covers this in full, including this project's actual embedding
model and the arithmetic behind "closest."

## 6. RAG as a pipeline, and where this project fits

Put together, a RAG system is a pipeline with distinct, separable stages:

```
documents  →  chunking  →  embedding  →  vector storage
                                                  │
question  →  embedding  →  similarity search  ←──┘
                                  │
                          retrieved passages
                                  │
                      LLM generates an answer
                    grounded in those passages
                                  │
                         (safety checks)
                                  │
                              response
```

This project (`enterprise-rag-platform`) implements every one of these stages as production code,
not a toy demo:

| Stage | This project's term | Chapter |
|---|---|---|
| Get text out of source files | Ingestion | [Ch 2](02-ingestion-and-chunking.md) |
| Split text into search-sized pieces | Chunking | [Ch 2](02-ingestion-and-chunking.md) |
| Turn text into vectors | Embedding | [Ch 3](03-embeddings-and-vector-search.md) |
| Store and search vectors | Vector store | [Ch 3](03-embeddings-and-vector-search.md) |
| Find the best-matching pieces | Retrieval + reranking | [Ch 4](04-retrieval-and-reranking.md) |
| Turn passages into a written answer | Generation | [Ch 5](05-generation-and-llms.md) |
| Catch PII leaks, hallucinations, unsafe input | Guardrails | [Ch 6](06-guardrails-and-safety.md) |
| Measure whether any of this actually works | Evaluation | [Ch 7](07-evaluation-framework.md) |

One more term worth defining now because it recurs constantly: **grounding**. An answer is
"grounded" when every claim in it traces back to something actually present in the retrieved
passages, as opposed to the model's own memorized (and possibly wrong) beliefs. Every design
decision from here on — the reranker, the prompt template, the hallucination detector — exists in
service of keeping answers grounded.

Next: [Chapter 1 — Project Overview & How to Run It](01-project-overview.md).
