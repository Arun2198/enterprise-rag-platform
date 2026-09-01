data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  count   = var.use_default_vpc ? 1 : 0
  default = true
}

data "aws_subnets" "default" {
  count = var.use_default_vpc ? 1 : 0

  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default[0].id]
  }
}

# Referenced, not created - see variables.tf's note on why IAM/Cognito
# stay outside this module's ownership.
data "aws_iam_role" "task_role" {
  name = var.existing_task_role_name
}

data "aws_iam_role" "task_execution_role" {
  name = var.existing_task_execution_role_name
}

# Referenced (validates the pool actually exists at plan/apply time),
# not created - same reasoning as the IAM roles above. Guarded by count
# rather than required so AUTH_ENABLED can still be turned off entirely
# by blanking existing_cognito_user_pool_id (see its own description).
data "aws_cognito_user_pool" "existing" {
  count        = var.existing_cognito_user_pool_id != "" ? 1 : 0
  user_pool_id = var.existing_cognito_user_pool_id
}

# The GitHub OIDC provider itself - referenced, not created. It's account-
# wide federation infrastructure that could plausibly back roles for
# other repos/workflows beyond this one; this module owns the deploy
# role that trusts it, not the provider itself.
data "aws_iam_openid_connect_provider" "github_actions" {
  url = "https://token.actions.githubusercontent.com"
}
