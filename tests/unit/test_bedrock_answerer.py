import json

from rag.chunking.chunk import Chunk
from rag.generation.bedrock_answerer import BedrockAnswerer
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class FakeBody:

    def read(self):
        return json.dumps(
            {
                "content": [
                    {
                        "text": "Grounded answer."
                    }
                ]
            }
        )


class FakeBedrockClient:

    def __init__(self):
        self.calls = []

    def invoke_model(self, **kwargs):
        self.calls.append(kwargs)
        return {"body": FakeBody()}


def test_bedrock_answerer_invokes_model_with_grounded_prompt():

    client = FakeBedrockClient()
    answerer = BedrockAnswerer(
        client=client,
        model_id="model-1"
    )
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="Only this context may be used.",
        source="doc.md",
        document_type="markdown",
    )
    retrieved = [
        RetrievedChunk(
            chunk=chunk,
            vector_score=0.8,
            keyword_score=0.5,
            score=0.7
        )
    ]

    answer = answerer.answer("What can be used?", retrieved)

    payload = json.loads(client.calls[0]["body"])
    prompt_text = payload["messages"][0]["content"][0]["text"]

    assert answer == "Grounded answer."
    assert "Only this context may be used." in prompt_text
    assert client.calls[0]["modelId"] == "model-1"
