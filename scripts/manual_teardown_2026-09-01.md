# Manual teardown — 2026-09-01 (local use only)

Why this exists: `provision-infra.yml -f action=destroy` (the automated
path) failed twice in a row for real, distinct reasons documented in
`CLOUD_NATIVE_GAPS.md`'s "CI-run terraform destroy is not reliable for
this module" section - once because destroy deleted the very IAM role it
was running as (now fixed with `prevent_destroy`), once because AWS
credentials expired ~12 minutes into the apply (OpenSearch domain
deletion alone takes that long; the underlying session-duration issue is
not yet fixed). This is the exact sequence used to finish the teardown
by hand once CI stopped being viable for it.

## 0. Recreate the deploy role CI had locked itself out of

Only needed because the first destroy attempt deleted
`github-actions-ecs-deploy-role` mid-run before `prevent_destroy` existed
in code. Not needed on a normal manual teardown once that fix is in
place - included here for completeness/reproducibility of what actually
happened.

```bash
cd terraform
source ../.env
export TF_VAR_jina_api_key="$JINA_API_KEY"
export TF_VAR_llm_api_key="$LLM_API_KEY"
export TF_VAR_container_image="849279003696.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag-platform:latest"

terraform plan \
  -target=aws_iam_role.github_actions_deploy \
  -target=aws_iam_role_policy_attachment.github_actions_ecr \
  -target=aws_iam_role_policy_attachment.github_actions_ec2 \
  -target=aws_iam_role_policy_attachment.github_actions_logs \
  -target=aws_iam_role_policy_attachment.github_actions_ecs \
  -target=aws_iam_role_policy_attachment.github_actions_elb \
  -target=aws_iam_role_policy_attachment.github_actions_s3 \
  -target=aws_iam_role_policy_attachment.github_actions_sqs \
  -target=aws_iam_role_policy_attachment.github_actions_secrets \
  -target=aws_iam_role_policy_attachment.github_actions_opensearch \
  -target=aws_iam_role_policy.github_actions_deploy_extras \
  -out=tfplan_recover_role.out

terraform apply "tfplan_recover_role.out"
```

## 1. Clear any stale state lock

A crashed CI run (or a killed local command) leaves the DynamoDB lock
held forever - the holder is gone and will never release it. Check the
lock ID in the error message, confirm nothing else is actually running,
then:

```bash
terraform force-unlock -force <lock-id>
```

Hit this twice in a row this session: once from the crashed CI run, once
from my own local `plan -destroy` getting killed by a 2-minute tool
timeout mid-refresh (OpenSearch/ECS refresh calls are slow).

## 2. Real destroy plan, from local credentials

```bash
cd terraform
source ../.env
export TF_VAR_jina_api_key="$JINA_API_KEY"
export TF_VAR_llm_api_key="$LLM_API_KEY"
export TF_VAR_container_image="849279003696.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag-platform:latest"

terraform plan -destroy -out=tfplan_manual_destroy.out
# review it, then:
terraform apply "tfplan_manual_destroy.out"
```

This alone got most of it - state correctly self-healed for anything
already gone from the earlier partial CI runs (OpenSearch domain, ECS
service, frontend bucket had already been destroyed before CI's
credentials died; `terraform plan -destroy`'s refresh step detected they
no longer existed and dropped them from state without erroring).

## 3. The `force_delete`/`force_destroy` gotcha

The ECR repo and docs bucket still failed with "not empty, consider
using force_delete" even though `terraform/ecr.tf` and `terraform/s3.tf`
already had `force_delete`/`force_destroy = true` committed. Real reason:
these are resource *attributes* stored in state, and a `-destroy`
operation reads whatever's already in state for them - it doesn't
reconcile config drift on other attributes first. Neither resource had
ever been through a normal (non-destroy) `apply` since that config
change landed, so state still had the old `false` value. Fixed with a
targeted normal apply first, to actually sync the attribute:

```bash
terraform apply -target=aws_ecr_repository.app -target=aws_s3_bucket.docs -auto-approve
```

Then destroy again - this time it actually force-deleted both:

```bash
terraform destroy -auto-approve
```

## 4. Verify

```bash
terraform state list   # should be empty

aws secretsmanager list-secrets --region us-east-1 --filters Key=name,Values=enterprise-rag-platform
aws ecr describe-repositories --region us-east-1
aws opensearch list-domain-names --region us-east-1
aws ecs list-clusters --region us-east-1
aws elbv2 describe-load-balancers --region us-east-1
aws sqs list-queues --region us-east-1 --queue-name-prefix enterprise-rag
aws scheduler list-schedules --region us-east-1
aws logs describe-log-groups --region us-east-1 --query "logGroups[?contains(logGroupName, 'enterprise-rag-platform')]"
```

All confirmed empty 2026-09-01. Only
`enterprise-rag-platform-tfstate-849279003696` (the Terraform state
bucket itself) and the DynamoDB lock table remain, correctly - those are
the backend, not app infrastructure, and were never meant to be torn
down by this.
