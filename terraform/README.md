# Terraform (Infrastructure as Code)

Reproducible IaC for this platform's AWS resources: ECR, S3 (docs +
frontend), SQS (+ DLQ, + a separate scheduler queue), EventBridge
Scheduler, OpenSearch, Secrets Manager, CloudWatch Logs, and a standard
ECS Fargate + ALB deployment.

## What this module does *not* manage, and why

- **The existing ECS task roles** (`ecsTaskRole`, `ecsTaskExecutionRole`) -
  referenced via `data` sources, not created. These already exist with
  the exact permissions this deployment needs (see the root `CLAUDE.md`
  and the deployment runbook), and IAM is free at rest - there's no cost
  benefit to Terraform owning them, and doing so would risk colliding
  with whatever the existing GitHub Actions pipeline already depends on.
  This module does create one genuinely new role
  (`scheduler_invocation`, in `scheduler.tf`) since EventBridge
  Scheduler has no existing role to assume - free at rest, same as any
  other IAM role.
- **Cognito User Pool** - never created by this module, only referenced (`data
  "aws_cognito_user_pool"`, validates it exists at plan/apply time) and reused
  via `existing_cognito_user_pool_id` (defaults to this project's real pool,
  `us-east-1_jkzIa7abx`). Recreating it would issue a new pool ID and
  invalidate every already-configured `OIDC_*` value across the app and its
  CI/CD pipeline. This is also what actually turns authentication on for the
  deployed app: `ecs.tf` sets `AUTH_ENABLED=true` plus the three `OIDC_*` env
  vars whenever the pool reference resolves - previously this module never set
  any of them, so the deployed ECS task always ran with auth off regardless of
  what the app itself supports. Blank `existing_cognito_user_pool_id` disables
  auth entirely rather than creating a fresh pool (this module doesn't do
  that - see the variable's own description).
- **The GitHub Actions OIDC provider / `github-actions-ecs-deploy-role`** -
  same reasoning as the task roles.

## Why standard ECS Fargate, not "ECS Express Mode"

The GitHub Actions deploy pipeline (`.github/workflows/deploy-aws.yml`)
uses `aws-actions/amazon-ecs-deploy-express-service`, a managed
convenience layer with no equivalent Terraform resource type as of this
module's authoring. This module provisions the *standard* equivalent
(`aws_ecs_service` + `aws_lb` + target group + listener) - same task
role, execution role, container port, and health check path, so the two
deployment paths are functionally interchangeable, but they are **not
the same resources** and applying this module does not adopt whatever
Express Mode created. See the root `CLAUDE.md` and the deployment
runbook for the Express Mode path, which remains the day-to-day
deployment method; this module is the from-scratch-reproducible
alternative.

HTTP only - no ACM certificate or custom domain is wired up. Add an
`aws_acm_certificate` + HTTPS listener before using this for anything
beyond staging/demo traffic.

## Cost summary (see each `.tf` file's own comments for detail)

| Resource | Cost while it exists |
|---|---|
| OpenSearch domain (t3.small.search) | **~$25-30/month** - the single largest cost driver, no free tier, runs continuously |
| ECS Fargate task (1024 CPU / 4096 MB, 1 task) | ~$40-45/month while `desired_count > 0` |
| Application Load Balancer | ~$16-20/month base charge, plus LCU under real traffic |
| Jina embeddings + reranking API (default provider, see below) | Per-call, usage-based - no fixed monthly charge, but now a real live third-party cost on every ingest and every query, not free local compute |
| S3, SQS, CloudWatch Logs, Secrets Manager, ECR | Negligible (a few cents/month combined at this project's scale) |

**Default embedding/reranking provider is Jina, not local models** (`EMBEDDING_PROVIDER=jina`/
`RERANKER_PROVIDER=jina` in `ecs.tf`) - a deliberate choice to satisfy the platform spec's
requirement that the AWS deployment be API-first rather than downloading models into the ECS
task. This trades the previous local-compute cost (CPU/memory already paid for in the Fargate
task) for a per-call API cost that scales with traffic - fine at this project's low query volume,
worth re-evaluating before any real production traffic. Requires `TF_VAR_jina_api_key` to be set
before applying; without it the ECS task fails at startup
(`ServiceConfigurationError: JINA_API_KEY not set`). Set `EMBEDDING_PROVIDER`/`RERANKER_PROVIDER`
back to their app-level defaults (`sentence_transformer`/`local`) in `ecs.tf` to revert to local
models if API cost or a live third-party dependency on every query is undesirable for a given
deployment.

Set `ecs_desired_count = 0` to stop paying for compute without destroying
the service definition. There's no equivalent "pause" for the OpenSearch
domain - it either exists (and bills) or is destroyed.

## Usage

```bash
cd terraform
terraform init
terraform plan -var-file=example.tfvars   # or your own terraform.tfvars
terraform apply -var-file=example.tfvars
```

First apply, before any image has been pushed to ECR: the ECS service
will fail to start tasks if `container_image` doesn't point at a real,
already-pushed image. Either apply everything except `aws_ecs_service`
first (`terraform apply -target=aws_ecr_repository.app ...`), push an
image via the existing CI/CD pipeline once the ECR repo exists, then
apply the rest - or set `ecs_desired_count = 0` for the first apply and
raise it once an image exists.

### After first apply

Update the GitHub Actions repo variables to point at this module's
outputs (`terraform output`) if you want the existing CI/CD pipeline to
target these Terraform-managed resources instead of - or alongside -
whatever it's currently pointed at. This module deliberately does not
overwrite `OPENSEARCH_HOST`/`S3_BUCKET`/`SQS_QUEUE_URL`/etc. itself.

### State

This module has no configured backend, so state is local
(`terraform.tfstate`, gitignored) - acceptable for a single-operator
account, not for a team. Add an `S3` + `DynamoDB` backend block in
`versions.tf` before more than one person runs `apply` against this
account.

## Validated, not yet applied

Every file here has been checked with `terraform validate` at the time
each `.tf` file was added, most recently after adding `scheduler.tf`
(EventBridge Scheduler + its SQS queue + IAM role) - `validate` passes
clean. The 24-resource `terraform plan` was run earlier against this
account's live IAM roles and default VPC before `scheduler.tf` existed,
which is what caught a real AWS-side constraint `terraform validate`
alone missed: OpenSearch domain names are capped at 28 characters, so
`opensearch_domain_name` is deliberately a separate, shorter variable
from `project_name`. `scheduler.tf` and the Cognito wiring in `data.tf`/
`ecs.tf` (the new `data "aws_cognito_user_pool"` reference and the
`AUTH_ENABLED`/`OIDC_*` env vars it feeds) have **not** yet been checked
against a real `terraform plan` (this account's own billed resources
were already torn down when they were written) - re-run `terraform plan`
against a real account before applying. Nothing here has been
**applied** - per this project's own cost-awareness rules, no one
should `apply` this without deciding to accept the ongoing cost above
first.
