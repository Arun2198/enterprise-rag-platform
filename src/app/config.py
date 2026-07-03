import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    vector_store_provider: str = "memory"
    embedding_provider: str = "hashing"
    generation_provider: str = "extractive"
    opensearch_host: str | None = None
    opensearch_index: str = "enterprise-rag-chunks"
    opensearch_username: str | None = None
    opensearch_password: str | None = None
    opensearch_use_ssl: bool = True
    opensearch_verify_certs: bool = True
    opensearch_timeout_seconds: int = 30
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"


def load_settings() -> Settings:
    return Settings(
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "memory"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing"),
        generation_provider=os.getenv("GENERATION_PROVIDER", "extractive"),
        opensearch_host=os.getenv("OPENSEARCH_HOST"),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "enterprise-rag-chunks"),
        opensearch_username=os.getenv("OPENSEARCH_USERNAME"),
        opensearch_password=os.getenv("OPENSEARCH_PASSWORD"),
        opensearch_use_ssl=_env_bool("OPENSEARCH_USE_SSL", default=True),
        opensearch_verify_certs=_env_bool("OPENSEARCH_VERIFY_CERTS", default=True),
        opensearch_timeout_seconds=int(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "30")),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ),
    )


def _env_bool(
    name: str,
    default: bool
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "y"}
