# The Enterprise RAG Platform Guide

This is the complete, from-scratch explanation of this project — what it is, why every piece
exists, and how it actually works, chapter by chapter. It's written for people who are new to
generative AI: nothing about embeddings, retrieval, LLMs, or RAG is assumed. Infrastructure
concepts (containers, cloud deployment, CI/CD) are also explained from the ground up.

This guide is a companion to [`CLAUDE.md`](../../CLAUDE.md) (the terse, practitioner-facing
architecture reference) and [`README.md`](../../README.md) (the quick-start). Those two assume
you already know what a RAG pipeline is. This guide assumes you don't, and builds up to the same
level of detail from zero.

**Every claim in this guide is checked against the actual source code as of the commit noted at
the bottom of each chapter** — not recalled from memory, not aspirational. If something in the
code changes, the chapter describing it needs to be re-verified, not just re-worded.

## How to read this

You can read start to finish (it's written to build on itself — Chapter 3 assumes you've read
Chapter 0's definition of an embedding, for example), or jump straight to whichever piece you
need. Each chapter is self-contained enough to make sense on its own if you already have the
prerequisite concepts from earlier chapters.

## Chapters

0. **[Introduction to GenAI, LLMs, and RAG](00-introduction-to-genai-and-rag.md)** — what a large
   language model actually is, what "hallucination" means and why it happens, and what problem
   Retrieval-Augmented Generation exists to solve. Start here if any of those terms are new.
1. **[Project Overview & How to Run It](01-project-overview.md)** — the whole system in one
   picture, the tech stack, and the exact commands to run the demo, the API, the tests, and the
   evaluation suite yourself.
2. **[Document Ingestion & Chunking](02-ingestion-and-chunking.md)** — how a PDF becomes text, and
   how that text gets split into the small pieces a RAG system actually searches over — including
   the real bug that was silently hurting this project's answer quality, and how it was found.
3. **[Embeddings & Vector Search](03-embeddings-and-vector-search.md)** — what it means to turn
   text into numbers so a computer can judge "similar meaning," and how this project stores and
   searches those numbers.
4. **[Retrieval & Reranking](04-retrieval-and-reranking.md)** — how the system decides which
   chunks are relevant to a question, combining old-school keyword matching with modern semantic
   search, then double-checking the results with a second, more careful model.
5. **[Generation: LLMs, Answerers, and Fallback](05-generation-and-llms.md)** — how the system
   actually turns retrieved chunks into a written answer, the three different ways this project
   can do that, and what happens when the primary one fails in production.
6. **[Guardrails & Safety](06-guardrails-and-safety.md)** — how the system checks itself for
   leaking private information, making things up, or being tricked, before a response ever reaches
   a user.
7. **[Evaluation Framework](07-evaluation-framework.md)** — how you actually measure whether a RAG
   system is any good, with real metrics and worked examples, not vibes.
8. **[MLOps Platform](08-mlops-platform.md)** — the operational backbone: versioning, promotion
   workflows, feature flags, scheduled jobs, audit trails, and backups.
9. **[Containers & Docker](09-containers-and-docker.md)** — what a container is and why this
   project is packaged as one, walked through this project's actual `Dockerfile` line by line.
10. **[AWS Cloud Deployment](10-aws-deployment.md)** — what cloud computing is, then every AWS
    service this project actually uses and why, including the real production incidents hit while
    deploying it.
11. **[CI/CD with GitHub Actions](11-cicd-and-github-actions.md)** — what continuous
    deployment is, and exactly what happens between a `git push` and a live, updated service.
12. **[Testing Strategy](12-testing-strategy.md)** — why and how this project tests itself without
    ever downloading a real AI model during a test run.
13. **[Security, Known Gaps, and Glossary](13-security-and-glossary.md)** — an honest accounting
    of what's deliberately not built yet, a real security fix walked through in detail, and a
    glossary of every term used across this guide.

## A note on honesty

This project is a genuine, working system, but it is not a finished, production-hardened product,
and this guide says so wherever that's true. Chapter 13 in particular lists what's missing on
purpose (like authentication) versus what's missing because it wasn't the priority yet. Treat
anything this guide describes as "not yet done" as still not done unless you've verified otherwise
in the current code.
