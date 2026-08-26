# Standard ECS Fargate + ALB, not "ECS Express Mode" - Express Mode is a
# newer, GitHub-Actions-integrated convenience layer with no mature
# Terraform resource type as of this module's authoring. This is the
# IaC-reproducible equivalent: same task role/execution role, same
# container port and health check, same autoscaling shape - just
# provisioned via the standard aws_ecs_service/aws_lb resources Terraform
# actually supports, rather than the managed Express Gateway API the
# GitHub Actions pipeline uses day-to-day. HTTP only (no ACM cert/custom
# domain wired up here) - add an aws_acm_certificate + HTTPS listener
# before using this for anything beyond staging/demo traffic.

resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-cluster"

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name = aws_ecs_cluster.this.name

  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    base              = 1
    weight            = 1
    capacity_provider = "FARGATE"
  }
}

resource "aws_security_group" "alb" {
  name        = "${var.project_name}-alb-sg"
  description = "Inbound HTTP from the internet to the ALB."
  vpc_id      = data.aws_vpc.default[0].id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_security_group" "ecs_tasks" {
  name        = "${var.project_name}-ecs-tasks-sg"
  description = "Inbound only from the ALB, on the app's container port."
  vpc_id      = data.aws_vpc.default[0].id

  ingress {
    description     = "From ALB"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Cost: an ALB has its own hourly charge (~$16-20/month) plus LCU charges
# under real traffic, independent of whether any Fargate task is running.
resource "aws_lb" "this" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.default[0].ids

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "app" {
  name        = "${var.project_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default[0].id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
    timeout             = 10
    matcher             = "200"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.this.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.project_name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = data.aws_iam_role.task_execution_role.arn
  task_role_arn            = data.aws_iam_role.task_role.arn

  container_definitions = jsonencode([
    {
      name      = var.project_name
      image     = var.container_image
      essential = true
      portMappings = [
        { containerPort = var.container_port, protocol = "tcp" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
      environment = concat(
        [
          { name = "RERANKER_ENABLED", value = "true" },
          # API-based, not a model downloaded into the ECS task - spec 1.4
          # requires this for the AWS deployment; sentence_transformer/local
          # remain the app's own bare-construction defaults for offline
          # local dev and tests, unaffected by this. Requires
          # TF_VAR_jina_api_key to be set before apply - see variables.tf.
          { name = "EMBEDDING_PROVIDER", value = "jina" },
          { name = "JINA_EMBEDDING_MODEL", value = var.jina_embedding_model },
          { name = "RERANKER_PROVIDER", value = "jina" },
          { name = "JINA_RERANK_MODEL", value = var.jina_rerank_model },
          { name = "EMBEDDING_MODEL_NAME", value = var.embedding_model_name },
          { name = "GENERATION_PROVIDER", value = var.generation_provider },
          { name = "GENERATION_FALLBACK_PROVIDER", value = var.generation_fallback_provider },
          { name = "AWS_REGION", value = var.aws_region },
          { name = "BEDROCK_MODEL_ID", value = var.bedrock_model_id },
          { name = "LLM_BASE_URL", value = var.llm_base_url },
          { name = "LLM_MODEL_NAME", value = var.llm_model_name },
          { name = "VECTOR_STORE_PROVIDER", value = "opensearch" },
          { name = "OPENSEARCH_HOST", value = aws_opensearch_domain.rag.endpoint },
          { name = "S3_BUCKET", value = aws_s3_bucket.docs.bucket },
          { name = "SQS_QUEUE_URL", value = aws_sqs_queue.ingestion.url },
          { name = "ASYNC_INGESTION_ENABLED", value = "true" },
          { name = "INGEST_ALLOWED_DIR", value = "sample_documents" },
          { name = "RETRIEVAL_RELEVANCE_GUARD_ENABLED", value = "true" },
          { name = "SCHEDULER_QUEUE_URL", value = aws_sqs_queue.scheduler.url },
          { name = "SCHEDULER_INTERVAL_SECONDS", value = tostring(var.scheduler_interval_minutes * 60) },
        ],
        var.cors_allowed_origin != "" ? [
          { name = "CORS_ALLOWED_ORIGINS", value = var.cors_allowed_origin }
        ] : []
      )
      secrets = concat(
        var.llm_api_key != "" ? [
          { name = "LLM_API_KEY", valueFrom = aws_secretsmanager_secret.llm_api_key[0].arn }
        ] : [],
        var.jina_api_key != "" ? [
          { name = "JINA_API_KEY", valueFrom = aws_secretsmanager_secret.jina_api_key[0].arn }
        ] : []
      )
    }
  ])

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_ecs_service" "app" {
  name            = "${var.project_name}-service"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.ecs_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default[0].ids
    security_groups  = [aws_security_group.ecs_tasks.id]
    assign_public_ip = true # avoids NAT Gateway cost - see variables.tf's use_default_vpc note
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.project_name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.http]

  lifecycle {
    ignore_changes = [task_definition] # CI/CD updates the running task
    # definition on every deploy - Terraform shouldn't fight that between
    # infrastructure-only applies.
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = var.ecs_max_count
  min_capacity       = var.ecs_desired_count
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu" {
  name               = "${var.project_name}-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70
  }
}
