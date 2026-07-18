import json
from unittest.mock import MagicMock
from unittest.mock import patch

from rag.chunking.chunk import Chunk
from rag.generation.bedrock_answerer import BedrockAnswerer
from rag.generation.openai_compatible_answerer import OpenAICompatibleAnswerer
from rag.generation.prompt import build_grounded_prompt
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


def _make_retrieved_chunk() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="Only this context may be used.",
        source="doc.md",
        document_type="markdown"
    )
    return RetrievedChunk(
        chunk=chunk,
        vector_score=0.8,
        keyword_score=0.5,
        score=0.7
    )


def _make_completion(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@patch("rag.generation.openai_compatible_answerer.OpenAI")
def test_bedrock_and_openai_compatible_share_identical_prompt(mock_openai_class):

    query = "What can be used?"
    retrieved = [_make_retrieved_chunk()]
    expected_prompt = build_grounded_prompt(query, retrieved)

    bedrock_client = FakeBedrockClient()
    BedrockAnswerer(
        client=bedrock_client,
        model_id="model-1"
    ).answer(query, retrieved)
    bedrock_payload = json.loads(bedrock_client.calls[0]["body"])
    bedrock_prompt = bedrock_payload["messages"][0]["content"][0]["text"]

    mock_client = mock_openai_class.return_value
    mock_client.chat.completions.create.return_value = _make_completion("Grounded answer.")
    OpenAICompatibleAnswerer(
        api_key="key",
        base_url="https://example.com/v1",
        model_name="gpt-4o-mini"
    ).answer(query, retrieved)
    openai_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]

    assert bedrock_prompt == expected_prompt
    assert openai_prompt == expected_prompt
