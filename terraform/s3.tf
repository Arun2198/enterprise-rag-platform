# Cost: negligible at this project's document volume - S3 standard
# storage is fractions of a cent per GB/month; the real cost driver would
# be request volume at production scale, not storage.
resource "aws_s3_bucket" "docs" {
  bucket = "${var.project_name}-docs-${data.aws_caller_identity.current.account_id}"
  # Without this, `terraform destroy` fails outright on a non-empty
  # bucket - found live during the first full teardown this session
  # (real ingested documents, job records, mlops backups all in here by
  # then), worked around by hand-emptying both buckets via `aws s3 rm
  # --recursive` before destroy could proceed. This makes that
  # unnecessary going forward - genuinely destructive on `destroy`, which
  # is exactly what was asked for both times this got hit.
  force_destroy = true

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "raw/processed/failed document storage + async ingestion job records"
  }
}

resource "aws_s3_bucket_public_access_block" "docs" {
  bucket                  = aws_s3_bucket.docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "docs" {
  bucket = aws_s3_bucket.docs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "frontend" {
  bucket = "${var.project_name}-frontend-${data.aws_caller_identity.current.account_id}"
  # Same reasoning as aws_s3_bucket.docs above.
  force_destroy = true

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "static frontend hosting"
  }
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket                  = aws_s3_bucket.frontend.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_policy" "frontend_public_read" {
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.frontend.arn}/*"
      }
    ]
  })

  depends_on = [aws_s3_bucket_public_access_block.frontend]
}
