# Cost: negligible - a few cents/month in image storage at this project's
# scale (a handful of images, not a high-churn registry).
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "MUTABLE"
  # Without this, `terraform destroy` fails outright on a non-empty repo
  # (ECR refuses to delete a repository that still has images) - found
  # live during the first full teardown this session, worked around by
  # hand-force-deleting the repo via the AWS CLI before destroy could
  # proceed. This makes that unnecessary going forward.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 7 days - keeps storage cost from growing unbounded across CI runs."
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      }
    ]
  })
}
