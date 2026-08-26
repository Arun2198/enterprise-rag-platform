# Chapter 8: MLOps Platform

Everything up to this point has been about answering questions well. This chapter is about
running the system as an actual *operated service*: how you roll out a change safely, how you
know who changed what, how you back up its state, and how you keep recurring jobs running. This is
`src/mlops/` — a provider-agnostic, dependency-free operational backbone.

## 1. What "MLOps" means here

"MLOps" (Machine Learning Operations) is the discipline of operating ML-backed systems reliably —
versioning models, controlling rollout risk, auditing changes, recovering from failure — the same
concerns "DevOps" covers for regular software, adapted for the specific wrinkles ML systems add
(a "deployment" might be a new model version or a new prompt template, not just new code; a
regression might be a quality metric drifting, not a crash).

Every component in `src/mlops/` is **in-memory by default with no persistence unless explicitly
backed up** — this is a library of building blocks, not a service with its own database. Most of
it (registries, lifecycle, governance) is available on `PlatformManager` but nothing in the live
app's request path writes to it yet. Two pieces *are* actively wired into the running app today:
**feature flags** and the **scheduler** — covered in section 5.

## 2. `PlatformManager` — the facade

`src/mlops/manager.py`. Every component below is independently usable on its own (`from
mlops.registry import ModelRegistry` needs no `PlatformManager` at all). `PlatformManager` exists
for two things: cross-component workflows (`promote()` does a lifecycle transition *and* mirrors
it into the governance audit log in one call, so the two can never drift out of sync), and
`register_provider(name, obj)`/`get_provider(name)` — one named slot any pluggable real backend
(an MLflow registry, a cloud secrets client, a CI pipeline, a drift detector) can register into
without `PlatformManager` needing to know its shape.

## 3. Registries — tracking versioned assets

- **`ModelRegistry`** (`registry.py`) — tracks versioned AI assets (embedding models, rerankers,
  LLM providers, prompt templates, guardrail models, evaluation models), keyed
  `"{asset_type}:{name}:{version}"`, each with a `LifecycleStage` and free-form metadata.
- **`ArtifactRegistry`** (`artifacts.py`) — separate, and **immutable/append-only**: prompt
  templates, chunking/embedding configs, eval datasets, experiment definitions, policies,
  guardrail configs. `save()` never overwrites an existing version — it always creates version
  `N+1`. This matters for reproducibility: you can always ask "what exact config produced this
  result," and the answer can't have been silently mutated after the fact.

## 4. Lifecycle — the promotion state machine

`lifecycle.LifecycleManager` (`lifecycle.py`) enforces a fixed promotion path for any registered
asset:

```
Development → Validation → Staging → Production → Retired
     ↑              ↓            ↓
     └──────────────┘            └── (Staging can also reject back to Validation)
```

The full legal-transition table lives in `ALLOWED_TRANSITIONS` — an asset can't jump straight
from `Development` to `Production`. Promoting into `Staging` or `Production` specifically requires
an `approved_by` argument (raises `ApprovalRequiredError` otherwise) — a deliberate human-in-the-
loop gate before anything reaches production. Every transition and every approval is recorded and
independently queryable via `.history(asset_id)` / `.approvals(asset_id)`.

## 5. What's actually wired into the live app

### Feature flags

`FeatureFlagManager` (`feature_flags.py`) supports boolean enable/disable, percentage-based canary
rollout, and a `shadow` marker (the flag only *tracks* shadow state; what "shadow mode" does with
that is entirely up to the caller's own code — this component doesn't implement shadow execution
itself).

The rollout mechanism, `is_enabled_for(name, subject_id)`, is worth understanding concretely,
since [Chapter 4](04-retrieval-and-reranking.md) relies on it for the reranker rollout:

```python
def is_enabled_for(self, name, subject_id):
    flag = self.get(name)
    if not flag.enabled:
        return False
    if flag.rollout_percentage >= 100.0:
        return True
    if flag.rollout_percentage <= 0.0:
        return False
    bucket = int(hashlib.sha256(f"{name}:{subject_id}".encode()).hexdigest(), 16) % 100
    return bucket < flag.rollout_percentage
```

Hashing `"{flag_name}:{subject_id}"` into a number `0-99` and comparing it against the rollout
percentage means the *same* subject always lands in the *same* bucket for a *given* flag — a user
at 25% rollout doesn't flip in and out between requests, because the hash of their id never
changes. This is the standard "stable canary bucketing" technique: deterministic without needing
to remember any per-user state.

This is genuinely live in the running app: `service_factory.build_rag_service()` attaches a
`FeatureFlagManager` (when `FEATURE_FLAGS_ENABLED=true`, the default) with a `cross_encoder_
reranker` flag pre-defined at `RERANKER_ROLLOUT_PERCENTAGE` (default `100`, i.e. unchanged
behavior). The admin endpoint `PATCH /admin/feature-flags/{name}` (in `app/main.py`) lets an
operator change `enabled`/`rollout_percentage` on a running service — and because `app/main.py`
passes the *same* `platform_manager` instance into both the admin routes and `build_rag_service()`,
flipping a flag via the API genuinely changes `/ask` behavior on the next request, not just a
disconnected copy of the flag.

### The scheduler

`Scheduler` (`scheduler.py`) is a real job registry with interval-based execution, but
**deliberately owns no background thread of its own** — you call `run_due_jobs(now)` periodically
from whatever actually owns scheduling in your deployment. This is a genuinely useful design
choice: it keeps the scheduler dependency-free and trivially testable with a fake `now` timestamp
instead of needing real sleeping/threading in tests, while leaving the actual "what drives the
clock" decision to the deployment context — a simple loop, a Kubernetes CronJob, a GitHub Actions
scheduled trigger, or plain cron could all drive the same `Scheduler` instance identically.

For the live FastAPI app, `app/main.py`'s `lifespan` context manager is that driver: it starts an
`asyncio` background task on app startup that sleeps for `SCHEDULER_INTERVAL_SECONDS` (default
300) and then calls `platform_manager.scheduler.run_due_jobs()`, in a loop, cancelled cleanly on
shutdown:

```python
async def _scheduler_loop() -> None:
    while True:
        await asyncio.sleep(settings.scheduler_interval_seconds)
        platform_manager.scheduler.run_due_jobs()
```

Two jobs are registered by default when the app starts: a `backup` job (calls
`PlatformManager.create_backup()` — snapshots the registry/artifacts/configuration/feature-flags
state to `SCHEDULER_BACKUP_DIR`, default `mlops_backups/`) and a `health_check` job (logs the
currently indexed chunk count). `POST /admin/scheduler/jobs/{job_id}/trigger` lets an operator run
either one immediately, outside its normal schedule.

## 6. The rest of the platform (built, tested, not yet wired into the request path)

These are real, working, independently-usable components — worth knowing about even though
nothing in `/ingest` or `/ask` calls them today:

- **Configuration** (`configuration.py`) — named, independently versioned environment profiles
  (dev/staging/prod), append-only like `ArtifactRegistry`. Optional per-key validators reject an
  entire `save_profile()` call if any value fails, so nothing partially invalid ever enters
  history. `rollback()` re-activates the previous version.
- **Secrets** (`secrets.py`) — a `SecretsProvider` Protocol plus `LocalEnvSecretsProvider` (reads
  `os.environ`, optionally namespaced by a prefix) as the only implemented backend. Every returned
  `SecretValue` prints `***redacted***` on `repr()`/`str()` — you must call `.reveal()` explicitly
  to get the raw string, so a secret can't accidentally end up in a log line via a bare `print()`
  or f-string. Cloud secret managers (Azure Key Vault, AWS Secrets Manager, GCP Secret Manager)
  are extension points, not implemented — no cloud SDK is a project dependency for this.
- **CI/CD** (`deployment.py`) — a `DeploymentPipeline` Protocol
  (`run_tests`/`run_evaluation`/`run_experiment`/`deploy`). `LocalDeploymentPipeline` genuinely
  shells out to `pytest` / `evaluation/run_eval.py` via `subprocess` — it's a real, working
  reference implementation, not a stub. `GitHubActionsDeploymentPipeline` implements the same
  Protocol against real GitHub Actions, driven through the `gh` CLI. See
  [Chapter 11](11-cicd-and-github-actions.md) for how the actually-used deployment pipeline (a
  plain GitHub Actions workflow file, not this Protocol) works.
- **Governance** (`governance.py`) — `GovernanceLog` is the audit trail: every
  register/promote/approve/lineage-link/policy-check call appends an `AuditEvent`, queryable by
  resource. `link_lineage(asset_id, artifact_id, version)` tracks which artifact versions produced
  a given model asset. `check_policy()` records the outcome either way — a passed check is exactly
  as visible in the log as a failed one.
- **Backup & recovery** (`backup.py`, `recovery.py`) — `BackupManager.create_snapshot({name:
  component, ...})` writes a timestamped local JSON file; anything with `.export_state()`/
  `.import_state()` qualifies. `RecoveryManager.restore_snapshot(path, components)` restores only
  the components explicitly named, silently skipping anything else present in the file. A cloud
  `BackupTarget` (S3/Azure Blob/GCS) is an extension point — this only ever writes locally today.
- **Permissions / RBAC** (`permissions.py`) — five roles (Administrator, MLEngineer, DataScientist,
  Reviewer, ReadOnly) checked against a static permission matrix
  (`has_permission`/`require_permission`). This only answers "given a role, is action X allowed" —
  establishing *who* the actor actually is (login, sessions, tokens) is out of scope by design and
  left to the caller. [Chapter 13](13-security-and-glossary.md) covers why this matters: the app
  has no authentication layer at all yet, so RBAC exists as a building block without anything
  upstream of it to feed it a real identity.
- **Drift & retraining** (`drift.py`, `retraining.py`) — Protocols only (`DriftDetector`,
  `RetrainingTrigger`, `ValidationWorkflow`); **not implemented**. No model training happens
  anywhere in this repo. A real `ValidationWorkflow` would naturally run an
  `evaluation.runner.EvaluationRunner` pass ([Chapter 7](07-evaluation-framework.md)) and gate on
  `evaluation.report.compare_reports`, then hand off to `LifecycleManager` for promotion — the
  pieces to build it already exist, they're just not connected yet.

## 7. Observability

`mlops/telemetry.py` follows the identical pattern to the guardrails telemetry from
[Chapter 6](06-guardrails-and-safety.md): OpenTelemetry API counters (`mlops.operations`,
`mlops.audit_events`), no-op with no `MeterProvider` configured, never raises on its own. Every
`PlatformManager` operation calls it alongside the matching `GovernanceLog` entry, so metrics and
the audit trail move together — you'll never see one without the other for the same event.

Next: [Chapter 9 — Containers & Docker](09-containers-and-docker.md).
