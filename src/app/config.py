import os
from dataclasses import dataclass


def _parse_bool(value: str) -> bool:
    return value.strip().lower() not in ("false", "0", "no", "")


@dataclass(frozen=True)
class Settings:
    vector_store_provider: str = "memory"
    embedding_provider: str = "hashing"
    generation_provider: str = "extractive"
    opensearch_host: str | None = None
    opensearch_index: str = "enterprise-rag-chunks"
    aws_region: str = "us-east-1"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model_name: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.0
    reranker_enabled: bool = True
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_candidate_multiplier: int = 4


def load_settings() -> Settings:
    return Settings(
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "memory"),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "hashing"),
        generation_provider=os.getenv("GENERATION_PROVIDER", "extractive"),
        opensearch_host=os.getenv("OPENSEARCH_HOST"),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", "enterprise-rag-chunks"),
        aws_region=os.getenv("AWS_REGION", "us-east-1"),
        bedrock_model_id=os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-haiku-20240307-v1:0"
        ),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o-mini"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1000")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        reranker_enabled=_parse_bool(os.getenv("RERANKER_ENABLED", "true")),
        reranker_model_name=os.getenv(
            "RERANKER_MODEL_NAME",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        reranker_candidate_multiplier=int(
            os.getenv("RERANKER_CANDIDATE_MULTIPLIER", "4")
        ),
    )
