from typing import Any

from rag.generation.prompt import build_grounded_prompt
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class BedrockAnswerer:
    """
    Bedrock runtime adapter with an injected boto3 bedrock-runtime client.
    Uses the Converse API (client.converse) rather than the raw
    invoke_model + provider-specific JSON body - Converse works uniformly
    across every Bedrock model provider (Anthropic, Amazon, Meta, Mistral,
    ...) and across inference profile ARNs (needed for some newer Claude
    models, which are only invocable via an inference profile rather than
    a plain model ID), so model_id can be either shape without this class
    caring which.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        max_tokens: int = 700,
        temperature: float = 0.0
    ) -> None:
        self.client = client
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk]
    ) -> str:
        if not retrieved_chunks:
            return "I could not find relevant context in the indexed documents."

        prompt = build_grounded_prompt(query, retrieved_chunks)
        response = self.client.converse(
            modelId=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature
            }
        )
        content = response.get("output", {}).get("message", {}).get("content", [])

        if content and "text" in content[0]:
            return content[0]["text"].strip()

        return ""
