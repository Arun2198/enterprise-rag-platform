# ADR-004 — Knowledge Domain Abstraction

## Status

Accepted

---

## Context

The platform must support multiple knowledge domains without requiring significant code changes.

Examples:

* AI Governance
* AWS Documentation
* Banking Regulations
* Enterprise Policies
* Healthcare Guidelines

---

## Decision

Introduce a Knowledge Domain abstraction.

A domain is represented through:

* Configuration
* Metadata
* Prompt Templates
* Evaluation Datasets

and not through platform code.

---

## Examples

AI Governance

domain = ai_governance

AWS

domain = aws

Banking

domain = banking

---

## Consequences

Positive

* High reusability
* Easier onboarding of new domains
* Reduced technical debt

Negative

* Additional configuration management

---

## Target

Switching domains should require less than 5% code change.