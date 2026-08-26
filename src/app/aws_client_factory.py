import boto3
from botocore.client import BaseClient
from botocore.config import Config


def build_boto3_client(
    service_name: str,
    region_name: str,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
    max_attempts: int = 3
) -> BaseClient:
    """
    Every boto3 client this app builds (Bedrock today; S3/SQS once wired
    into the live app) should go through this rather than a bare
    boto3.client(...) call - botocore's own defaults for timeouts and
    retry behavior vary by version and aren't something to rely on
    implicitly. "standard" retry mode retries throttling/5xx/connection
    errors with exponential backoff automatically; connect/read timeouts
    stop a single hung request from blocking a request thread forever.
    """
    config = Config(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        retries={"max_attempts": max_attempts, "mode": "standard"}
    )
    return boto3.client(service_name, region_name=region_name, config=config)
