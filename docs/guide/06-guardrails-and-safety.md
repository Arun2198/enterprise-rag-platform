# Chapter 6: Guardrails & Safety

Generation (Chapter 5) produces an answer. Before that answer ever reaches a user, it passes
through **guardrails**: automated checks for things like leaked private data or ungrounded
(hallucinated) claims. This chapter covers `src/rag/guardrails/`.

## 1. The `Guardrail` interface

Every guardrail — simple regex-based ones and ML-model-backed ones alike — implements one shape
(`src/rag/guardrails/base.py`):

```python
class Guardrail(Protocol):
    name: str
    stage: GuardrailStage       # INPUT or OUTPUT
    def check(self, context: GuardrailContext) -> GuardrailFinding: ...
```

A guardrail declares which **stage** it runs at: `INPUT` (checks the user's question, before any
retrieval happens) or `OUTPUT` (checks the generated answer, after generation). `GuardrailContext`
carries whatever a check needs — `query` always, and `answer` + `retrieved_chunks` for
output-stage checks. A `GuardrailFinding` reports whether the guardrail triggered, how severe that
is, and what action it recommends:

```python
class Severity(str, Enum):
    INFO = "info"; WARNING = "warning"; HIGH = "high"; CRITICAL = "critical"

class Action(str, Enum):
    ALLOW = "allow"; WARN = "warn"; REDACT = "redact"; ESCALATE = "escalate"; BLOCK = "block"
```

Actions have a strict severity ordering (`ALLOW < WARN < REDACT < ESCALATE < BLOCK`). This
matters because a request can trigger several guardrails at once — the system needs one final
decision, not several conflicting ones.

## 2. `GuardrailManager` — running the checks and picking a verdict

`src/rag/guardrails/manager.py`. `RAGService.ask()` calls it twice per request:

```python
input_result = self.guardrail_manager.run_input(query)          # before retrieval
if input_result.action == Action.BLOCK:
    return AskResponse(answer=input_result.text, sources=[], ...)   # short-circuit
...
output_result = self.guardrail_manager.run_output(query, answer, retrieved_chunks)  # after generation
if output_result.action == Action.BLOCK:
    return AskResponse(answer=output_result.text, sources=[], ...)  # never leaks retrieved text
```

A `BLOCK` on the input stage means retrieval and generation never even run — no wasted LLM call
for a request that was going to be rejected anyway. A `BLOCK` on the output stage means the
*retrieved chunk text never leaves the system* — `sources` comes back empty and the answer is
replaced with a fixed message (`BLOCKED_MESSAGE`), because if the answer itself is unsafe, the raw
context it was built from might be too.

Internally, `_run()`:

1. Runs every guardrail registered for the current stage, in order.
2. If a guardrail's finding includes `redacted_text` (only `PIIGuard` does this today), the text
   is replaced *before* the next guardrail in the sequence sees it — so if you had two guardrails
   both scanning output text, the second one sees the redacted version, not the raw one.
3. Resolves one final `Action` as the *maximum* severity across every triggered finding
   (`_resolve_action`) — one guardrail warning and another blocking means the overall result is a
   block, never averaged or overridden by the milder one.
4. `_build_flags()` turns findings into `AskResponse.guardrail_flags` — `pii_detected` and
   `hallucination`/`groundedness` get pulled out into their own top-level fields (matching the
   project's HLD-specified response shape), and every finding, including from any additional
   guardrail with no special-cased field, also appears in a generic `details` list — so adding a
   new guardrail never requires a schema change for it to show up in the API response.

## 3. Phase 1 defaults: `PIIGuard` and `HallucinationDetector`

`GuardrailManager.default()` (used when `RAGService` isn't given a manager explicitly) and
`service_factory._build_guardrail_manager()` (the live app's real wiring) both register these two
output-stage guardrails by default.

### `PIIGuard` — regex PII redaction

`src/rag/guardrails/pii_guard.py`. Scans the answer text for five entity patterns — email, phone,
SSN, credit card, Aadhaar (Indian national ID) — each independently toggleable, each with its own
severity (credit card is `CRITICAL`, email is only `INFO`). Detected entities are replaced
in-place with a placeholder like `[REDACTED_EMAIL]`, and the finding's `redacted_text` carries the
cleaned version forward. Patterns are checked in a fixed, most-specific-first order (`EMAIL, SSN,
CREDIT_CARD, AADHAAR, PHONE`) so a looser pattern like `PHONE` never "eats" digits that a more
specific pattern like `SSN` already matched in the same pass.

Worked example: an LLM answer contains `"contact jane@company.com for details, SSN 123-45-6789"`.
`PIIGuard.check()` detects `EMAIL` and `SSN`, redacts to `"contact [REDACTED_EMAIL] for details,
SSN [REDACTED_SSN]"`, sets `action=REDACT`, and severity is the max of the two entities' severities
(`SSN` is `HIGH`, which wins over `EMAIL`'s `INFO`). This redacted text is what actually reaches
the user.

### `HallucinationDetector` — groundedness scoring

`src/rag/guardrails/hallucination_detector.py`. Estimates how much of the generated answer is
actually supported by the retrieved context, blending two signals:

- **Token overlap**: what fraction of the answer's words also appear somewhere in the retrieved
  chunks' combined text. Cheap, has zero dependencies, but blind to paraphrase (an answer that
  correctly *rephrases* the context scores artificially low).
- **Embedding cosine similarity** (Chapter 3) between the answer's embedding and the combined
  context's embedding, when an `Embedder` is available — `RAGService` always has one
  (`HashingEmbedder` by default for direct construction, the real sentence-transformer model for
  the live app), and this reuses that same instance rather than constructing a separate one — so
  this signal comes "for free," with groundedness scoring staying consistent with whatever
  embedding space retrieval is actually using.

```python
blended = 0.6 * token_overlap + 0.4 * cosine_similarity   # clamped to [0, 1]
likely_hallucination = blended < threshold   # default threshold 0.60
```

Below threshold, it triggers with `action=WARN` (never `BLOCK` on its own in the Phase 1 default
set — a low groundedness score is a *signal*, not proof of a fabricated answer, so the default
response is to flag it via `guardrail_flags.hallucination`/`.groundedness` rather than hide the
answer outright). `GROUNDEDNESS_THRESHOLD` is the tuning knob.

## 4. Optional guardrails — same interface, not on by default

Three more dependency-free guardrails exist and implement the exact same `Guardrail` interface,
but aren't in the Phase 1 default set — each needs to be registered explicitly to opt in:

- **`PromptInjectionGuard`** (input stage, `src/rag/guardrails/prompt_injection_guard.py`) — regex
  heuristics for jailbreak/injection phrasing ("ignore previous instructions," etc.) in the user's
  question.
- **`SecretLeakageGuard`** (output stage, `src/rag/guardrails/secret_leakage_guard.py`) —
  API-key/token/private-key pattern matching on generated answers, so the LLM can't accidentally
  echo something that looks like a credential back to a user.
- **`ProfanityGuard`** (output stage, `src/rag/guardrails/profanity_guard.py`) — a small
  illustrative wordlist check.

Three more, ML-backed and heavier, also exist and are wired only when their respective
`*_ENABLED` setting is turned on in `service_factory._build_guardrail_manager()`:

- **`PresidioPIIGuard`** (`presidio_pii_guard.py`, `PRESIDIO_PII_GUARD_ENABLED`) — uses Microsoft
  Presidio's NER model (plus a spaCy `en_core_web_sm` model, loaded once at construction) instead
  of plain regex, catching context-dependent PII like names and addresses that a fixed pattern
  structurally can't recognize. Registered *alongside* `PIIGuard`, not replacing it — it adds
  coverage the regex guard doesn't have (including a custom Aadhaar recognizer, since Presidio has
  no built-in one), while `PIIGuard` still handles the exact-pattern entities cheaply. Overlapping
  detections (e.g. a URL entity fully inside an email match) are resolved by keeping the
  higher-confidence, longer span.
- **`NLIHallucinationDetector`** (`nli_hallucination_detector.py`, `NLI_HALLUCINATION_ENABLED`,
  default model `cross-encoder/nli-deberta-v3-base`) — a stronger groundedness check using
  **Natural Language Inference**: for each retrieved chunk, it asks a model "does this chunk
  entail this answer" (the same premise/hypothesis framing NLI models are trained for) and takes
  the *maximum* entailment probability across chunks as the groundedness score — the answer only
  needs to be supported by at least one chunk, not all of them.
- **`LLMJudgeHallucinationDetector`** (`llm_judge_hallucination_detector.py`, `LLM_JUDGE_ENABLED`)
  — asks an LLM itself to score groundedness via a JSON-only prompt (reusing
  `OpenAICompatibleAnswerer`'s client/retry pattern). **Fails open**: any API error or unparseable
  response means this guardrail simply doesn't trigger (never blocks) and marks
  `metadata["judge_available"] = False` — an unreachable judge should never become an outage for
  the whole guardrails stage.

Toxicity/hate-speech classification, BERTScore, and RAGAS-style metrics are **not implemented** —
documented as a deliberate choice, not an oversight: a low-quality toxicity classifier can cause
real harm and was judged to deserve dedicated attention rather than a quick add, and
BERTScore/RAGAS were judged to mostly duplicate what the NLI and LLM-judge detectors already
cover here.

## 5. `PolicyEngine` — escalating, never downgrading

`src/rag/guardrails/policy.py`. A `PolicyEngine` evaluates configurable rules (a guardrail name +
minimum severity, and/or a metadata threshold) that can *escalate* the action a `GuardrailManager`
would otherwise take — for example, "if PII severity is `CRITICAL`, escalate to `BLOCK` even
though `PIIGuard`'s own default action is only `REDACT`." It can never *downgrade* an action a
guardrail already decided on. It's opt-in — pass `policy_engine=` into `GuardrailManager`
explicitly; by default, Phase 1 findings apply their own suggested action directly (PII redacts,
hallucination warns, neither auto-blocks on its own).

## 6. Observability

`src/rag/guardrails/telemetry.py`. Every guardrail check and every resolved action gets recorded
via OpenTelemetry API counters/histograms (`guardrail.runs`, `guardrail.latency`,
`guardrail.pii_detections`, `guardrail.hallucination_flags`, `guardrail.groundedness_score`,
`guardrail.blocked_responses`). With no `MeterProvider` configured (the default — this repo
doesn't ship a Prometheus/Grafana setup), these calls are cheap no-ops; a host application can plug
in a real exporter later and start getting these metrics retroactively with zero code changes on
this side. Recording is wrapped so it can never raise — a broken metrics exporter must not be able
to break the guardrail pipeline it's supposed to be observing.

## 7. Tracing a blocked response end to end

Continuing the running example: suppose a retrieved chunk happens to contain a stray email
address, and the LLM's answer echoes it back verbatim as `"reach out to admin@company.com for
approval"`.

1. `RAGService.ask()` calls `run_output(query, answer, retrieved_chunks)`.
2. `PIIGuard.check()` finds `EMAIL`, sets `action=REDACT`, `redacted_text="reach out to
   [REDACTED_EMAIL] for approval"`.
3. `HallucinationDetector.check()` runs against the *redacted* text (since `PIIGuard` ran first
   and its redaction updated the shared context) and, say, scores groundedness at `0.82` — above
   threshold, so it doesn't trigger.
4. `_resolve_action()` sees one triggered finding (`REDACT`) and no `BLOCK`, so the overall action
   is `REDACT`.
5. `RAGService.ask()` returns the redacted answer, real `sources`, and `guardrail_flags =
   {"pii_detected": true, "hallucination": false, "groundedness": 0.82, "details": [...]}`.

If instead the hallucination score had come back at, say, `0.30`, the overall action would still
be `REDACT` (the max of `REDACT` and `WARN`) — the flags would additionally show
`"hallucination": true`, surfacing the low-confidence signal to the caller without hiding the
answer.

Next: [Chapter 7 — Evaluation Framework](07-evaluation-framework.md).
