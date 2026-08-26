# Cost: negligible at this project's log volume - CloudWatch Logs pricing
# is per-GB ingested/stored; a 14-day retention window keeps storage cost
# from growing unbounded, which is the actual risk factor at scale.
resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}"
  retention_in_days = 14

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
