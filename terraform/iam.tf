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
