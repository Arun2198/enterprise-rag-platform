import json
from typing import Any

from rag.generation.prompt import build_grounded_prompt
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class BedrockAnswerer:
    """
    Bedrock runtime adapter with an injected boto3 bedrock-runtime client.
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
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(self._claude_messages_payload(prompt)),
            contentType="application/json",
            accept="application/json"
        )
        body = response.get("body")
        payload = json.loads(body.read() if hasattr(body, "read") else body)
        content = payload.get("content", [])

        if content and "text" in content[0]:
            return content[0]["text"].strip()

        return ""

    def _claude_messages_payload(
        self,
        prompt: str
    ) -> dict[str, Any]:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
