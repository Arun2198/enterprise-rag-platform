terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Was local-file state before this - meant a killed apply once left a
  # real resource (the OpenSearch domain) untracked until manually
  # `terraform import`ed back, and made running apply from CI structurally
  # unsafe (an ephemeral runner has no local state to persist between
  # runs - every CI apply would start from blank state and try to
  # recreate everything). The S3 bucket + DynamoDB table below are a
  # one-time bootstrap (see scripts/bootstrap_terraform_state.md) - they
  # can't be created by the Terraform they back, so they're provisioned
  # once by hand, outside this module, the one genuinely unavoidable
  # manual step in this whole pipeline.
  backend "s3" {
    bucket         = "enterprise-rag-platform-tfstate-849279003696"
    key            = "enterprise-rag-platform/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "enterprise-rag-platform-tflock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region
}
