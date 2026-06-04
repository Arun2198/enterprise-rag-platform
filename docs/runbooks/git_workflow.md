# Git Workflow

## Branching Strategy

Main Branch

main

Purpose:

* Production-ready code
* Stable documentation
* Approved architecture

---

Feature Branches

Naming Convention:

feature/<module>

Examples:

feature/ingestion

feature/chunking

feature/embeddings

feature/retrieval

feature/generation

feature/evaluation

---

Documentation Branches

Naming Convention:

docs/<topic>

Examples:

docs/ingestion-lld

docs/chunking-lld

---

Commit Convention

Format:

<type>: <description>

Examples:

docs: add ingestion LLD

docs: add ADR for parser strategy

feat: implement PDF parser

feat: add recursive chunker

test: add parser unit tests

refactor: improve chunk metadata propagation

---

Merge Policy

A feature branch can only be merged when:

* LLD exists
* ADR exists
* Implementation complete
* Unit tests pass
* Documentation updated

---

Pull Request Template

Summary

What was implemented?

Testing

What was tested?

Documentation

What documentation was updated?

Risks

Any known risks?

---

Golden Rule

No direct commits to main after initialization.
