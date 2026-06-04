# ADR-003 — Domain Agnostic Platform Design

## Status

Accepted

---

## Context

The platform must support multiple knowledge domains.

Examples:

* AI Governance
* AWS Documentation
* Banking Regulations
* Enterprise Policies

Without abstraction, domain-specific logic would spread throughout the codebase.

---

## Decision

The platform will remain domain agnostic.

All domain-specific behavior will be expressed through:

* Configuration
* Metadata
* Prompt Templates
* Evaluation Datasets

---

## Consequences

Positive

* High reusability
* Lower maintenance
* Faster onboarding

Negative

* Additional configuration management

---

## Success Metric

Switching domains should require less than 5% code change.