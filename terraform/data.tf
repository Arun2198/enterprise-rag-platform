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
