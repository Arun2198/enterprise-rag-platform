# MASTER ROADMAP

## Project

Enterprise Knowledge Intelligence Platform

### Reference Implementation

AI Governance & Risk Intelligence System

### Reference Dataset

* NIST AI Risk Management Framework (AI RMF 1.0)
* NIST AI RMF Playbook
* NIST GenAI Profile

### Design Principles

* Platform first, use case second
* Domain agnostic architecture
* Configuration driven behavior
* Code Reuse Target >= 95%
* Version-aware knowledge management

### Future Supported Domains

* AI Governance
* AWS Documentation
* Banking Regulations
* Enterprise Policies
* Healthcare Guidelines
* Internal Knowledge Bases

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

# PHASE 0.5 — DOMAIN ABSTRACTION

Goal

Create a domain-agnostic platform architecture.

Modules

LLD-000 Domain Abstraction

Artifacts

* Knowledge Domain Abstraction
* Domain Configuration Strategy
* Metadata Strategy
* Prompt Template Strategy

Configuration

configs/

```
domains/

    ai_governance.yaml
```

ADR

ADR-003 Domain Agnostic Platform Design

ADR-004 Knowledge Domain Abstraction

Outcome

New domains can be onboarded with minimal code changes.

Status

IN PROGRESS

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
* Markdown Parser

Tests

* Valid PDF
* Corrupt PDF
* Empty PDF
* Large PDF

ADR

ADR-001 Parser Framework Design

Status

IN PROGRESS

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

IN PROGRESS

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
* Embedding Model Comparison

ADR

ADR-005 Embedding Model Selection

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

ADR-006 OpenSearch Selection

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

ADR-007 Retrieval Strategy

Status

NOT STARTED

---

# PHASE 5.5 — VERSION MANAGEMENT

Goal

Support evolving knowledge bases.

Artifacts

* Document Version Tracking
* Change Detection
* Incremental Re-indexing
* Version Metadata

Experiments

* Retrieval Across Versions
* Change Impact Analysis

ADR

ADR-008 Version Aware Knowledge Management

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

ADR-009 Prompting Strategy

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
* POST /compare-versions

ADR

ADR-010 API Design

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

ADR-011 Hybrid Retrieval

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

ADR-012 Reranking Strategy

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
* Context Relevance

ADR

ADR-013 Evaluation Framework

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

ADR-014 Feedback Collection

Status

NOT STARTED

---

# PHASE 12 — RECOMMENDATION ENGINE

Goal

Recommend related knowledge.

Artifacts

* Similar Documents
* Related Knowledge Artifacts
* Cross Referenced Sections
* Recommendation Ranking

ADR

ADR-015 Recommendation Strategy

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

ADR-016 Observability Design

Status

NOT STARTED

---

# PHASE 14 — DRIFT DETECTION

Goal

Detect retrieval degradation and knowledge evolution.

Artifacts

* Embedding Drift
* Query Drift
* Quality Drift
* Knowledge Drift

ADR

ADR-017 Drift Detection

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

ADR-018 Experimentation Framework

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

ADR-019 Deployment Strategy

Status

NOT STARTED

---

# PHASE 17 — DOMAIN ONBOARDING

Goal

Demonstrate platform extensibility.

Artifacts

* AI Governance Domain
* AWS Documentation Domain
* Banking Regulations Domain

Deliverables

* Domain Configuration Templates
* Metadata Mapping Templates
* Evaluation Dataset Templates

Success Criteria

A new domain can be onboarded without modifying core platform code.

ADR

ADR-020 Domain Onboarding Framework

Status

NOT STARTED