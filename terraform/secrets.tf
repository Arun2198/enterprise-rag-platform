# Cost: ~$0.40/month flat fee per secret, plus negligible API-call cost.
# Only created when a real key is supplied - var.llm_api_key defaults to
# empty precisely so `terraform apply` never accidentally creates a
# secret holding an empty string.
resource "aws_secretsmanager_secret" "llm_api_key" {
  count = var.llm_api_key != "" ? 1 : 0

  name        = "${var.project_name}/llm-api-key"
  description = "API key for the openai_compatible fallback generation provider."

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "llm_api_key" {
  count = var.llm_api_key != "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.llm_api_key[0].id
  secret_string = var.llm_api_key
}

resource "aws_iam_role_policy" "task_execution_secrets_access" {
  count = var.llm_api_key != "" ? 1 : 0

  name = "${var.project_name}-secrets-access"
  role = data.aws_iam_role.task_execution_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "${aws_secretsmanager_secret.llm_api_key[0].arn}*"
      }
    ]
  })
}
