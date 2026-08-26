# Copy to terraform.tfvars (gitignored) and fill in real values, or pass
# individual -var flags / TF_VAR_* env vars instead. Never commit a file
# containing a real llm_api_key.

aws_region   = "us-east-1"
project_name = "enterprise-rag-platform"
environment  = "staging"

# Must point at a real, already-pushed image - see container_image's
# description in variables.tf. Leave the ECS-affecting resources out of
# your first apply (see README.md) until this exists.
container_image = "849279003696.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag-platform:latest"

generation_provider          = "bedrock"
generation_fallback_provider = "openai_compatible"
llm_base_url                 = "https://integrate.api.nvidia.com/v1"
llm_model_name               = "meta/llama-3.1-8b-instruct"
# llm_api_key = "set via TF_VAR_llm_api_key, never in this file"

# Required before applying with the default EMBEDDING_PROVIDER=jina /
# RERANKER_PROVIDER=jina (ecs.tf) - without it the ECS task fails at
# startup (ServiceConfigurationError: JINA_API_KEY not set).
# jina_api_key = "set via TF_VAR_jina_api_key, never in this file"

cors_allowed_origin = "http://enterprise-rag-platform-frontend-849279003696.s3-website-us-east-1.amazonaws.com"
