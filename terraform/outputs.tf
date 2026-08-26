output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "opensearch_endpoint" {
  value = aws_opensearch_domain.rag.endpoint
}

output "s3_docs_bucket" {
  value = aws_s3_bucket.docs.bucket
}

output "s3_frontend_website_endpoint" {
  value = aws_s3_bucket_website_configuration.frontend.website_endpoint
}

output "sqs_queue_url" {
  value = aws_sqs_queue.ingestion.url
}

output "scheduler_queue_url" {
  value = aws_sqs_queue.scheduler.url
}

output "alb_dns_name" {
  description = "Public URL for the deployed API (HTTP only - see ecs.tf's note on adding HTTPS)."
  value       = aws_lb.this.dns_name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "ecs_service_name" {
  value = aws_ecs_service.app.name
}
