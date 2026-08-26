# Cost: effectively free - well within the SQS always-free tier
# (1M requests/month) at this project's ingestion volume.
resource "aws_sqs_queue" "ingestion_dlq" {
  name                      = "${var.project_name}-ingestion-dlq"
  message_retention_seconds = 1209600 # 14 days - max retention, so a failed
  # message survives long enough to be investigated before it's gone for good.

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_sqs_queue" "ingestion" {
  name                       = "${var.project_name}-ingestion-queue"
  visibility_timeout_seconds = 120
  message_retention_seconds  = 345600 # 4 days

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingestion_dlq.arn
    maxReceiveCount     = 5
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}
