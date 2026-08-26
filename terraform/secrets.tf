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

# Cost: same ~$0.40/month flat fee as llm_api_key, only created when a real
# key is supplied. This is what makes EMBEDDING_PROVIDER=jina /
# RERANKER_PROVIDER=jina (ecs.tf) actually work in the deployed task -
# spec 1.4 requires the AWS deployment to use API-based embedding/
# reranking rather than downloading local models into the ECS task, and
# this is the credential that path needs.
resource "aws_secretsmanager_secret" "jina_api_key" {
  count = var.jina_api_key != "" ? 1 : 0

  name        = "${var.project_name}/jina-api-key"
  description = "API key for Jina embedding + reranking (the AWS deployment's default providers)."

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_secretsmanager_secret_version" "jina_api_key" {
  count = var.jina_api_key != "" ? 1 : 0

  secret_id     = aws_secretsmanager_secret.jina_api_key[0].id
  secret_string = var.jina_api_key
}

resource "aws_iam_role_policy" "task_execution_jina_secrets_access" {
  count = var.jina_api_key != "" ? 1 : 0

  name = "${var.project_name}-jina-secrets-access"
  role = data.aws_iam_role.task_execution_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "${aws_secretsmanager_secret.jina_api_key[0].arn}*"
      }
    ]
  })
}
