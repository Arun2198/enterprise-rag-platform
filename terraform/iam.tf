# Grants ecsTaskRole (the running container's own AWS calls) access to
# the actual resources THIS module creates. Free at rest, same as any
# other IAM policy.
#
# Before this existed, ecsTaskRole only had a hand-maintained inline
# policy (enterprise-rag-live-backends, created manually in an earlier
# session) scoped to a *previous* deployment's queue name
# (enterprise-rag-ingestion-queue). This module names its SQS queue
# enterprise-rag-platform-ingestion-queue (prefixed by project_name) -
# a different resource entirely, so a fresh deployment through this
# module got real, live AccessDenied errors on both the SQS enqueue in
# POST /documents and the background SQS ingestion worker's own poll
# loop, discovered by actually uploading a real file through the
# deployed app (real S3 write, then unauthorized sqs:SendMessage).
#
# This resource makes correct permissions part of the reproducible
# apply, not a manual step someone has to remember after every fresh
# deployment.
# --- GitHub Actions deploy role ---------------------------------------
# Unlike ecsTaskRole/ecsTaskExecutionRole/Cognito above, this role has no
# other consumer to protect by staying hands-off - it exists for exactly
# one purpose (deploy-aws.yml assuming it), so it's fully Terraform-owned
# here: trust policy, managed-policy attachments, and inline policy are
# all real resources, imported from the role that was originally
# bootstrapped by hand before this module existed (see
# `terraform import aws_iam_role.github_actions_deploy
# github-actions-ecs-deploy-role` in the deployment runbook) rather than
# recreated. A drifted or hand-edited permission on this role (like the
# S3 write access below, added by hand via `aws iam put-role-policy`
# before this existed) is now visible in `terraform plan` instead of
# invisible outside the module entirely.
resource "aws_iam_role" "github_actions_deploy" {
  name = var.existing_github_actions_deploy_role_name

  # A real, serious failure found live: `terraform destroy` runs AS this
  # role's assumed credentials. Without this, destroy deletes the role
  # itself partway through its own run, instantly invalidating its own
  # in-flight session - every subsequent parallel resource-destroy call
  # in that same apply started failing with a mix of
  # UnrecognizedClientException/AccessDeniedException/InvalidAccessKeyId
  # (all symptoms of the exact same underlying cause: the credentials'
  # issuing role no longer existed), state failed to persist to S3 for
  # the same reason, and the next CI run couldn't even authenticate
  # ("Could not assume role with OIDC: Not authorized to perform
  # sts:AssumeRoleWithWebIdentity") since there was no role left to
  # assume. Recovery required local, non-role-dependent AWS credentials -
  # CI had zero path back in on its own. prevent_destroy makes this
  # structurally impossible to hit again: any plan that would destroy
  # this resource now errors instead of executing.
  lifecycle {
    prevent_destroy = true
  }

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Federated = data.aws_iam_openid_connect_provider.github_actions.arn
        }
        Action = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:${var.github_repository}:*"
          }
        }
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# The broad AWS-managed policies this role already had attached before
# being brought under Terraform - kept as-is (not narrowed) since this
# apply's job is to make existing state visible and reproducible, not to
# silently change what CI can do. AmazonEC2FullAccess/CloudWatchLogsFullAccess
# are real over-grants for what this role actually needs (image
# build/push, ECS deploy, frontend S3 sync) - flagged here rather than
# fixed here, since narrowing them is a real behavior change that
# deserves its own deliberate pass, not a side effect of an IAM-visibility
# fix.
resource "aws_iam_role_policy_attachment" "github_actions_ecr" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_ec2" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2FullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_logs" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_ecs" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_elb" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

# Added when provision-infra.yml (a real `terraform apply` from CI) first
# needed to create OpenSearch/S3/SQS/SecretsManager resources, not just
# redeploy onto existing ones - same broad-managed-policy style as the
# five above, same explicit non-goal of narrowing anything in this pass.
resource "aws_iam_role_policy_attachment" "github_actions_s3" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_sqs" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSQSFullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_secrets" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/SecretsManagerReadWrite"
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_iam_role_policy_attachment" "github_actions_opensearch" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonOpenSearchServiceFullAccess"
  lifecycle {
    prevent_destroy = true
  }
}

# Same inline policy name ("ecs-deploy-extras") the role already carried -
# Terraform now owns its content going forward. iam:PassRole lets the
# deploy workflow hand these roles to the ECS service/tasks it creates;
# application-autoscaling:* supports the autoscaling target this module
# provisions. FrontendDeploy, TerraformManagedIamRoles,
# CognitoDescribeForDataSource, EventBridgeScheduler,
# EcrRepositoryManagement, and TerraformStateLock were all added by hand
# via `aws iam put-role-policy` before this resource existed to track
# them - TerraformStateLock specifically was found missing live: the
# very first `terraform apply` run from CI failed acquiring the state
# lock (DynamoDB GetItem/PutItem AccessDenied) because granting S3 access
# to the state bucket doesn't imply DynamoDB access to the lock table -
# two separate services, two separate grants needed.
resource "aws_iam_role_policy" "github_actions_deploy_extras" {
  name = "ecs-deploy-extras"
  role = aws_iam_role.github_actions_deploy.name

  lifecycle {
    prevent_destroy = true
  }

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Missing the scheduler-invocation role until this was found live:
        # aws_scheduler_schedule needs iam:PassRole on the role it hands
        # to EventBridge Scheduler (scheduler_invocation below), same as
        # ECS needs it for the task/execution roles - a from-scratch
        # apply failed creating both schedules over this.
        #
        # aws_iam_role.scheduler_invocation.arn is deliberately NOT used
        # here (same reasoning as FrontendDeploy's literal ARN below) -
        # referencing the resource attribute made this policy depend on
        # scheduler_invocation, so any -target destroy plan that included
        # it pulled this protected, prevent_destroy'd resource in too.
        # Literal ARN instead - the role name is deterministic.
        Sid    = "PassEcsRoles"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          data.aws_iam_role.task_execution_role.arn,
          "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/ecsInfrastructureRoleForExpressServices",
          data.aws_iam_role.task_role.arn,
          "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-scheduler-invocation-role"
        ]
      },
      {
        Sid      = "AppAutoscaling"
        Effect   = "Allow"
        Action   = "application-autoscaling:*"
        Resource = "*"
      },
      {
        # Deliberately a literal ARN, not aws_s3_bucket.frontend.arn - a
        # real bug found live: referencing the resource attribute makes
        # this policy structurally depend on the frontend bucket, so any
        # -target destroy plan that includes the bucket pulls this
        # protected, prevent_destroy'd resource in too ("Instance cannot
        # be destroyed" on github_actions_deploy_extras, even though it
        # was correctly excluded from -target directly) - the exact same
        # dependency-based circular-inclusion pattern as the OIDC
        # provider ARN fix elsewhere in this file, just discovered later.
        # The bucket name is deterministic (project_name +
        # account_id, both known without needing the resource itself).
        Sid      = "FrontendDeploy"
        Effect   = "Allow"
        Action   = ["s3:PutObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project_name}-frontend-${data.aws_caller_identity.current.account_id}",
          "arn:aws:s3:::${var.project_name}-frontend-${data.aws_caller_identity.current.account_id}/*"
        ]
      },
      {
        # Includes its own ARN, not just the roles it manages elsewhere -
        # found missing on the second CI-run import attempt: reading and
        # updating itself (GetRole to refresh state, UpdateAssumeRolePolicy
        # for its own trust policy, AttachRolePolicy/DetachRolePolicy for
        # its own managed-policy attachments) needs the exact same
        # permissions as managing any other role, and a role doesn't
        # implicitly have any of these on itself - IAM requires an
        # explicit grant even for self-management.
        Sid    = "TerraformManagedIamRoles"
        Effect = "Allow"
        Action = [
          "iam:GetRole",
          "iam:CreateRole",
          "iam:DeleteRole",
          "iam:TagRole",
          "iam:UntagRole",
          "iam:UpdateAssumeRolePolicy",
          "iam:PutRolePolicy",
          "iam:GetRolePolicy",
          "iam:DeleteRolePolicy",
          "iam:ListRolePolicies",
          "iam:ListAttachedRolePolicies",
          "iam:ListInstanceProfilesForRole",
          "iam:AttachRolePolicy",
          "iam:DetachRolePolicy"
        ]
        Resource = [
          data.aws_iam_role.task_role.arn,
          data.aws_iam_role.task_execution_role.arn,
          "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/${var.project_name}-scheduler-invocation-role",
          aws_iam_role.github_actions_deploy.arn
        ]
      },
      {
        Sid      = "CognitoDescribeForDataSource"
        Effect   = "Allow"
        Action   = "cognito-idp:DescribeUserPool"
        Resource = "arn:aws:cognito-idp:${var.aws_region}:${data.aws_caller_identity.current.account_id}:userpool/${var.existing_cognito_user_pool_id}"
      },
      {
        Sid    = "EventBridgeScheduler"
        Effect = "Allow"
        Action = [
          "scheduler:CreateSchedule",
          "scheduler:DeleteSchedule",
          "scheduler:GetSchedule",
          "scheduler:UpdateSchedule",
          "scheduler:TagResource",
          "scheduler:ListTagsForResource",
          "scheduler:ListSchedules"
        ]
        Resource = "*"
      },
      {
        Sid    = "EcrRepositoryManagement"
        Effect = "Allow"
        Action = [
          "ecr:CreateRepository",
          "ecr:DeleteRepository",
          "ecr:TagResource",
          "ecr:PutLifecyclePolicy",
          "ecr:GetLifecyclePolicy",
          "ecr:DeleteLifecyclePolicy",
          "ecr:PutImageScanningConfiguration"
        ]
        Resource = "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/${var.project_name}"
      },
      {
        Sid      = "TerraformStateLock"
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:DeleteItem"]
        Resource = "arn:aws:dynamodb:${var.aws_region}:${data.aws_caller_identity.current.account_id}:table/${var.project_name}-tflock"
      },
      {
        # Found missing on the very first CI-run terraform import: this
        # role manages its own trust policy (aws_iam_role.github_actions_deploy
        # above, which references data.aws_iam_openid_connect_provider),
        # a genuine bootstrap wrinkle - it needs to read the OIDC
        # provider that grants it access in the first place.
        # ListOpenIDConnectProviders has no per-resource ARN form (it's
        # an account-wide list operation), so it's the one statement here
        # that can't be scoped tighter than "*".
        Sid      = "GithubOidcProviderRead"
        Effect   = "Allow"
        Action   = "iam:ListOpenIDConnectProviders"
        Resource = "*"
      },
      {
        # Deliberately NOT data.aws_iam_openid_connect_provider.github_actions.arn
        # here - that would make this policy depend on successfully
        # reading the very data source this permission exists to allow
        # reading, a real circular dependency that broke the first CI
        # apply attempt (AccessDenied evaluating the data source before
        # this policy granting access to it had ever been applied).
        # Literal ARN instead - GitHub's OIDC provider URL/thumbprint
        # namespace this hangs off of doesn't change.
        Sid      = "GithubOidcProviderGet"
        Effect   = "Allow"
        Action   = "iam:GetOpenIDConnectProvider"
        Resource = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"
      }
    ]
  })
}


resource "aws_iam_role_policy" "task_role_live_backends" {
  name = "${var.project_name}-task-role-live-backends"
  role = data.aws_iam_role.task_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "OpenSearchAccess"
        Effect = "Allow"
        Action = [
          "es:ESHttpGet",
          "es:ESHttpPost",
          "es:ESHttpPut",
          "es:ESHttpDelete",
          "es:ESHttpHead"
        ]
        Resource = "${aws_opensearch_domain.rag.arn}/*"
      },
      {
        Sid    = "S3DocumentStoreAccess"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = "${aws_s3_bucket.docs.arn}/*"
      },
      {
        Sid      = "S3BucketList"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.docs.arn
      },
      {
        Sid    = "SQSIngestionAccess"
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.ingestion.arn
      }
    ]
  })
}
