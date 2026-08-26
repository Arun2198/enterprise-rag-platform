variable "aws_region" {
  description = "AWS region for every resource this module creates."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix applied to every resource name - keeps this module's resources distinct from anything hand-created outside Terraform, so apply/destroy never collides with resources it doesn't own."
  type        = string
  default     = "enterprise-rag-platform"
}

variable "environment" {
  description = "Deployment environment tag (dev/staging/production) - purely descriptive, doesn't branch any resource logic in this module."
  type        = string
  default     = "staging"
}

# --- OpenSearch -------------------------------------------------------
# Real, ongoing cost driver - the single largest line item. t3.small.search
# single-node is the cheapest configuration that supports k-NN vector
# search; it has no free tier. Estimated ~$25-30/month for the instance
# plus ~$1/month for 10GB gp3 storage, running continuously.
variable "opensearch_instance_type" {
  description = "OpenSearch data node instance type."
  type        = string
  default     = "t3.small.search"
}

variable "opensearch_instance_count" {
  description = "Number of OpenSearch data nodes. 1 = no high availability, cheapest option, fine for staging/dev."
  type        = number
  default     = 1
}

variable "opensearch_volume_size_gb" {
  description = "EBS gp3 volume size per OpenSearch node, in GB."
  type        = number
  default     = 10
}

variable "opensearch_engine_version" {
  description = "OpenSearch engine version."
  type        = string
  default     = "OpenSearch_2.19"
}

variable "opensearch_domain_name" {
  description = "OpenSearch domain names are capped at 28 characters by AWS, so this is intentionally separate from project_name (which is too long once '-search' is appended)."
  type        = string
  default     = "enterprise-rag-search"

  validation {
    condition     = length(var.opensearch_domain_name) <= 28
    error_message = "OpenSearch domain names must be 28 characters or fewer."
  }
}

# --- ECS Fargate --------------------------------------------------------
# Real, ongoing cost driver - Fargate compute plus the ALB's own hourly
# charge run continuously whenever desired_count > 0. At the defaults
# below (1024 CPU / 4096 MB, 1 task), expect roughly $40-45/month in
# Fargate compute plus ~$16-20/month for the ALB, before any traffic.
variable "ecs_task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU, 1024 = 1 vCPU, ...)."
  type        = string
  default     = "1024"
}

variable "ecs_task_memory" {
  description = "Fargate task memory in MB. This app loads embedding/reranker models into memory - keep at 4096 or higher."
  type        = string
  default     = "4096"
}

variable "ecs_desired_count" {
  description = "Number of running tasks. Set to 0 to stop paying for compute without destroying the service definition."
  type        = number
  default     = 1
}

variable "ecs_max_count" {
  description = "Upper bound for autoscaling."
  type        = number
  default     = 2
}

variable "container_image" {
  description = "Full ECR image URI (repository:tag) to deploy. Left blank by default since the image doesn't exist until the CI pipeline has built and pushed at least once - set this explicitly before the first apply that creates the ECS service, or the service will fail to start tasks."
  type        = string
  default     = ""
}

variable "container_port" {
  description = "Port the FastAPI app listens on inside the container."
  type        = number
  default     = 8000
}

# --- Networking -----------------------------------------------------
variable "use_default_vpc" {
  description = "Use the account's default VPC/subnets instead of creating new networking. Avoids NAT Gateway cost entirely (a real, meaningful cost driver per the platform's own cost-awareness rules) by running Fargate tasks with public IPs directly in public subnets, matching what this project has actually run in practice."
  type        = bool
  default     = true
}

# --- S3 / SQS -------------------------------------------------------
variable "s3_max_file_size_mb" {
  description = "Passed through as an app env var - not an S3-level enforcement, just documents the app's own upload limit alongside the bucket that receives it."
  type        = number
  default     = 25
}

# --- Scheduling -------------------------------------------------------
variable "scheduler_interval_minutes" {
  description = "How often EventBridge Scheduler fires each registered job (backup, health_check). Matches the app's own SCHEDULER_INTERVAL_SECONDS default (300s = 5 minutes) - EventBridge Scheduler's rate expression only supports whole-minute granularity, hence minutes here rather than seconds."
  type        = number
  default     = 5
}

# --- Application configuration (passed to the ECS task as env vars) -
variable "embedding_model_name" {
  type    = string
  default = "BAAI/bge-base-en-v1.5"
}

variable "generation_provider" {
  type        = string
  default     = "bedrock"
  description = "extractive | bedrock | openai_compatible"
}

variable "generation_fallback_provider" {
  type        = string
  default     = "openai_compatible"
  description = "Empty string disables the fallback provider entirely."
}

variable "bedrock_model_id" {
  type    = string
  default = "arn:aws:bedrock:us-east-1:849279003696:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"
}

variable "llm_base_url" {
  description = "Base URL for the openai_compatible fallback provider."
  type        = string
  default     = "https://integrate.api.nvidia.com/v1"
}

variable "llm_model_name" {
  type    = string
  default = "meta/llama-3.1-8b-instruct"
}

variable "llm_api_key" {
  description = "API key for the openai_compatible fallback provider. Marked sensitive - never commit a real value; pass via TF_VAR_llm_api_key or a .tfvars file that's gitignored. Empty string means no secret is created and GENERATION_FALLBACK_PROVIDER should be left unset."
  type        = string
  default     = ""
  sensitive   = true
}

variable "cors_allowed_origin" {
  description = "Origin allowed to call the API from a browser (the frontend's own URL). Empty string leaves CORS disabled entirely, matching the app's own default."
  type        = string
  default     = ""
}

# --- Existing resources this module deliberately does NOT create ----
# IAM is free at rest and these roles already exist with the exact
# permissions this deployment needs (see CLAUDE.md's Wiring section and
# the deployment runbook) - referenced via data sources in iam.tf rather
# than recreated, so this module can be applied without disturbing
# whatever's already using them (e.g. the existing GitHub Actions
# pipeline). Cognito is similarly referenced, not recreated, since
# recreating it would issue a new user pool ID and invalidate every
# already-configured OIDC_* value.
variable "existing_task_role_name" {
  type    = string
  default = "ecsTaskRole"
}

variable "existing_task_execution_role_name" {
  type    = string
  default = "ecsTaskExecutionRole"
}

variable "existing_cognito_user_pool_id" {
  description = "Set to the existing Cognito User Pool ID to reuse it (recommended - avoids invalidating OIDC_* config elsewhere). Leave blank to have this module create a fresh pool instead."
  type        = string
  default     = ""
}
