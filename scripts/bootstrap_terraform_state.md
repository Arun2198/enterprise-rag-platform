# Terraform remote state bootstrap

One-time, run once by hand, outside any pipeline - the S3 bucket and
DynamoDB table below are what Terraform's own state depends on, so they
structurally can't be created by a `terraform apply` that needs them to
already exist. This is not a gap to automate; every real Terraform+CI
setup has exactly this one unavoidable manual step, done once.

Already run for this project (2026-09-01) - this file exists so it's
reproducible if the account/region ever needs a fresh backend, not
because it needs running again.

```bash
STATE_BUCKET="enterprise-rag-platform-tfstate-849279003696"
LOCK_TABLE="enterprise-rag-platform-tflock"

aws s3api create-bucket --bucket "$STATE_BUCKET" --region us-east-1

aws s3api put-bucket-versioning \
  --bucket "$STATE_BUCKET" \
  --versioning-configuration Status=Enabled \
  --region us-east-1

aws s3api put-bucket-encryption \
  --bucket "$STATE_BUCKET" \
  --region us-east-1 \
  --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block \
  --bucket "$STATE_BUCKET" \
  --region us-east-1 \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws dynamodb create-table \
  --table-name "$LOCK_TABLE" \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1
```

Then `terraform/versions.tf`'s `backend "s3"` block points at these by
name. After creating them (or if you ever need to re-point at a fresh
pair), run `terraform init` (or `terraform init -reconfigure` if
switching backends) to pick up the backend config.

Cost: negligible - the state file is a few KB, versioning keeps old
copies but they're tiny too; DynamoDB on-demand billing for lock
acquire/release calls (one pair per `apply`/`plan`) is well within the
free tier at this project's usage.
