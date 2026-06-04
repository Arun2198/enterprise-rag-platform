# MASTER ROADMAP

## Project

Enterprise Knowledge Intelligence Platform

---

# PHASE 0 — PROJECT INITIALIZATION

Status: COMPLETE

Deliverables

* Documentation structure
* HLD
* Project Journal
* Technology decisions

Outcome

* Single source of truth established

---

# PHASE 1 — FOUNDATION

Goal

Create core data contracts and ingestion framework.

Modules

LLD-001 Ingestion

Artifacts

* Document Contract
* Parser Interfaces
* PDF Parser
* DOCX Parser

Tests

* Valid PDF
* Corrupt PDF
* Empty PDF
* Large PDF

ADR

ADR-001 Parser Framework Design

Status

NOT STARTED

---

# PHASE 2 — CHUNKING

Goal

Transform documents into retrieval-ready chunks.

Modules

LLD-002 Chunking

Artifacts

* Chunk Contract
* Recursive Chunker
* Metadata Propagation

Experiments

* Chunk Size Comparison
* Chunk Overlap Comparison

ADR

ADR-002 Chunking Strategy

Status

NOT STARTED

---

# PHASE 3 — EMBEDDINGS

Goal

Transform chunks into vector representations.

Modules

LLD-003 Embedding Pipeline

Artifacts

* Embedding Contract
* BGE Integration
* Embedding Service

Experiments

* BGE-small Baseline

ADR

ADR-003 Embedding Model Selection

Status

NOT STARTED

---

# PHASE 4 — INDEXING

Goal

Store vectors and metadata.

Modules

LLD-004 Indexing

Artifacts

* OpenSearch Index
* Index Management
* Metadata Storage

ADR

ADR-004 OpenSearch Selection

Status

NOT STARTED

---

# PHASE 5 — RETRIEVAL

Goal

Retrieve relevant knowledge.

Modules

LLD-005 Retrieval

Artifacts

* Vector Retrieval
* Top-K Selection

Experiments

* Top-K Optimization

ADR

ADR-005 Retrieval Strategy

Status

NOT STARTED

---

# PHASE 6 — GENERATION

Goal

Generate grounded responses.

Modules

LLD-006 Generation

Artifacts

* Prompt Builder
* Context Injection
* Citation Generation

ADR

ADR-006 Prompting Strategy

Status

NOT STARTED

---

# PHASE 7 — API LAYER

Goal

Expose platform capabilities.

Modules

LLD-007 FastAPI Layer

Endpoints

* POST /ingest
* POST /ask

ADR

ADR-007 API Design

Status

NOT STARTED

---

# PHASE 8 — HYBRID RETRIEVAL

Goal

Combine semantic and keyword retrieval.

Artifacts

* BM25
* Hybrid Fusion

Experiments

* Vector vs Hybrid

ADR

ADR-008 Hybrid Retrieval

Status

NOT STARTED

---

# PHASE 9 — RERANKING

Goal

Improve retrieval quality.

Artifacts

* Cross Encoder
* Reranking Service

Experiments

* Hybrid vs Hybrid + Reranker

ADR

ADR-009 Reranking Strategy

Status

NOT STARTED

---

# PHASE 10 — EVALUATION

Goal

Measure retrieval and generation quality.

Artifacts

* Golden Dataset
* Recall@K
* Precision@K
* MRR
* Faithfulness

ADR

ADR-010 Evaluation Framework

Status

NOT STARTED

---

# PHASE 11 — FEEDBACK SYSTEM

Goal

Collect user feedback.

Artifacts

* Thumbs Up
* Thumbs Down
* Feedback Store

ADR

ADR-011 Feedback Collection

Status

NOT STARTED

---

# PHASE 12 — RECOMMENDATION ENGINE

Goal

Recommend related content.

Artifacts

* Similar Documents
* Related Policies
* Recommendation Ranking

ADR

ADR-012 Recommendation Strategy

Status

NOT STARTED

---

# PHASE 13 — OBSERVABILITY

Goal

Measure platform behavior.

Artifacts

* Query Logs
* Cost Tracking
* Latency Tracking

ADR

ADR-013 Observability Design

Status

NOT STARTED

---

# PHASE 14 — DRIFT DETECTION

Goal

Detect retrieval degradation.

Artifacts

* Embedding Drift
* Query Drift
* Quality Drift

ADR

ADR-014 Drift Detection

Status

NOT STARTED

---

# PHASE 15 — EXPERIMENTATION

Goal

Safely evolve the platform.

Artifacts

* A/B Testing
* Traffic Splitting
* Promotion Rules
* Rollback Rules

ADR

ADR-015 Experimentation Framework

Status

NOT STARTED

---

# PHASE 16 — DEPLOYMENT

Goal

Production deployment.

Artifacts

* Docker
* CI/CD
* Monitoring

ADR

ADR-016 Deployment Strategy

Status

NOT STARTED
