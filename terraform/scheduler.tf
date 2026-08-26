# Fixes a real bug in the plain in-process interval scheduler: with
# ecs_max_count > 1 (autoscaling can run 2 tasks), each task independently
# fires every registered job on its own asyncio interval loop, so a
# "backup"/"health_check" job runs twice per interval instead of once -
# there's no coordination between tasks holding separate in-memory
# Scheduler instances. Routing the trigger through SQS fixes this for
# free (SQS delivers each message to exactly one consumer at a time
# regardless of how many tasks are polling), the same single-delivery
# guarantee the ingestion queue already relies on. See
# mlops/sqs_scheduler_worker.py.
#
# Cost: SQS is effectively free at this volume (well within the 1M
# requests/month free tier); EventBridge Scheduler has no separate
# per-schedule charge, only a negligible per-invocation cost at 2
# schedules firing every 5 minutes (~576 invocations/month combined).

resource "aws_sqs_queue" "scheduler" {
  name                       = "${var.project_name}-scheduler-queue"
  visibility_timeout_seconds = 60
  message_retention_seconds  = 3600 # 1 hour - a stale scheduled-job trigger
  # (backup/health_check) isn't worth replaying long after the fact; the
  # next EventBridge-scheduled run will fire again on its own cron anyway.

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role" "scheduler_invocation" {
  name = "${var.project_name}-scheduler-invocation-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "scheduler.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_iam_role_policy" "scheduler_send_to_sqs" {
  name = "${var.project_name}-scheduler-send-to-sqs"
  role = aws_iam_role.scheduler_invocation.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sqs:SendMessage"]
        Resource = aws_sqs_queue.scheduler.arn
      }
    ]
  })
}

# The running app polls this queue (mlops.sqs_scheduler_worker), so its
# task role - already referenced via data.aws_iam_role.task_role
# elsewhere in this module - needs receive/delete on it too.
resource "aws_iam_role_policy" "task_scheduler_queue_access" {
  name = "${var.project_name}-scheduler-queue-access"
  role = data.aws_iam_role.task_role.name

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.scheduler.arn
      }
    ]
  })
}

resource "aws_scheduler_schedule" "backup" {
  name       = "${var.project_name}-backup"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = "rate(${var.scheduler_interval_minutes} minutes)"

  target {
    arn      = aws_sqs_queue.scheduler.arn
    role_arn = aws_iam_role.scheduler_invocation.arn
    input    = jsonencode({ job_id = "backup" })
  }
}

resource "aws_scheduler_schedule" "health_check" {
  name       = "${var.project_name}-health-check"
  group_name = "default"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = "rate(${var.scheduler_interval_minutes} minutes)"

  target {
    arn      = aws_sqs_queue.scheduler.arn
    role_arn = aws_iam_role.scheduler_invocation.arn
    input    = jsonencode({ job_id = "health_check" })
  }
}
