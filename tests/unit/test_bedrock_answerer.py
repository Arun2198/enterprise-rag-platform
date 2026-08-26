from rag.chunking.chunk import Chunk
from rag.generation.bedrock_answerer import BedrockAnswerer
from rag.retrieval.hybrid_retrieval import RetrievedChunk


class FakeBedrockClient:

    def __init__(self):
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Grounded answer."}]
                }
            }
        }


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

    prompt_text = client.calls[0]["messages"][0]["content"][0]["text"]

    assert answer == "Grounded answer."
    assert "Only this context may be used." in prompt_text
    assert client.calls[0]["modelId"] == "model-1"


def test_bedrock_answerer_passes_inference_config():

    client = FakeBedrockClient()
    answerer = BedrockAnswerer(
        client=client,
        model_id="model-1",
        max_tokens=500,
        temperature=0.2
    )
    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="context",
        source="doc.md",
        document_type="markdown",
    )
    retrieved = [RetrievedChunk(chunk=chunk, vector_score=0.5, keyword_score=0.5, score=0.5)]

    answerer.answer("q", retrieved)

    assert client.calls[0]["inferenceConfig"] == {"maxTokens": 500, "temperature": 0.2}


def test_bedrock_answerer_returns_fallback_message_when_no_chunks_retrieved():

    client = FakeBedrockClient()
    answerer = BedrockAnswerer(client=client, model_id="model-1")

    answer = answerer.answer("q", [])

    assert answer == "I could not find relevant context in the indexed documents."
    assert client.calls == []


def test_bedrock_answerer_returns_empty_string_when_response_has_no_content():

    class EmptyContentClient:
        def converse(self, **kwargs):
            return {"output": {"message": {"role": "assistant", "content": []}}}

    chunk = Chunk(
        chunk_id="doc:0",
        document_id="doc",
        chunk_index=0,
        text="context",
        source="doc.md",
        document_type="markdown",
    )
    retrieved = [RetrievedChunk(chunk=chunk, vector_score=0.5, keyword_score=0.5, score=0.5)]

    answerer = BedrockAnswerer(client=EmptyContentClient(), model_id="model-1")

    assert answerer.answer("q", retrieved) == ""


def test_bedrock_answerer_records_generation_telemetry_when_usage_present():
    from conftest import TELEMETRY_READER

    class FakeBedrockClientWithUsage:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"role": "assistant", "content": [{"text": "Answer."}]}},
                "usage": {"inputTokens": 120, "outputTokens": 40, "totalTokens": 160}
            }

    answerer = BedrockAnswerer(
        client=FakeBedrockClientWithUsage(),
        model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )
    chunk = Chunk(
        chunk_id="doc:0", document_id="doc", chunk_index=0,
        text="Some context.", source="doc.md", document_type="md"
    )
    retrieved = [RetrievedChunk(chunk=chunk, vector_score=0.9, keyword_score=0.0, score=0.9)]

    answerer.answer("a question", retrieved)

    data = TELEMETRY_READER.get_metrics_data()
    input_token_points = []
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == "generation.input_tokens":
                    input_token_points.extend(metric.data.data_points)

    matching = [
        p for p in input_token_points
        if p.attributes.get("model") == "anthropic.claude-3-haiku-20240307-v1:0"
    ]
    assert matching
    assert matching[0].value >= 120
