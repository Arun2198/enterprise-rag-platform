# Chunking Strategy — Interview Story

## Problem

Enterprise documents are often large and cannot be directly embedded or retrieved efficiently.

A chunking strategy is required to transform documents into retrieval-ready units while preserving context.

---

## Alternatives Considered

### Fixed Length Chunking

Pros

* Simple

Cons

* Breaks context
* Poor retrieval quality

---

### Semantic Chunking

Pros

* Better context preservation

Cons

* Higher complexity
* Increased processing cost

---

### Rule-Based Chunking

Pros

* Respects document structure

Cons

* Domain dependent

---

## Decision

Selected Recursive Chunking for Phase 1.

Reasons:

* Deterministic
* Easy to evaluate
* Strong baseline
* Supports future experimentation

---

## Tradeoffs

Advantages

* Simplicity
* Predictability
* Faster implementation

Disadvantages

* Reduced semantic awareness
* Potential context fragmentation

---

## Expected Impact

Benefits:

* Improved retrieval consistency
* Easier experimentation
* Faster implementation

Future experiments will compare Recursive Chunking against Semantic Chunking and Rule-Based Chunking.