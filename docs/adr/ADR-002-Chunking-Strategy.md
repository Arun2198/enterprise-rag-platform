# ADR-002 — Chunking Strategy

## Status

Accepted

---

## Context

Chunking is one of the most important components in a Retrieval-Augmented Generation system.

Poor chunking directly impacts:

* Embedding quality
* Retrieval quality
* Answer quality

The platform requires a chunking strategy that is simple, reliable, and extensible.

---

## Decision

Phase 1 will use Recursive Chunking.

---

## Rationale

Benefits:

* Easy to implement
* Easy to test
* Deterministic
* Good baseline for evaluation

Recursive Chunking provides a strong foundation before introducing more advanced techniques.

---

## Alternatives Considered

### Fixed Length Chunking

Rejected.

Reasons:

* Ignores document structure
* Higher context fragmentation

---

### Semantic Chunking

Deferred.

Reasons:

* Higher complexity
* Requires additional evaluation framework

Will be introduced in Phase 2.

---

### Rule-Based Chunking

Deferred.

Reasons:

* Domain specific
* Requires custom parsing logic

Will be introduced in future phases.

---

## Consequences

Positive

* Faster implementation
* Easier debugging
* Predictable behavior

Negative

* Less context awareness
* Potential semantic boundary loss

---

## Future Considerations

Future chunking experiments:

* Recursive vs Semantic
* Recursive vs Rule-Based
* Different chunk sizes
* Different overlap values

These experiments will be tracked through the experimentation framework.