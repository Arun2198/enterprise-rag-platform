from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ingest_and_ask_endpoints(tmp_path):

    file_path = tmp_path / "leave_policy.md"
    file_path.write_text(
        "Employees receive 20 days of paid leave annually. "
        "Contractors receive 10 days of leave.",
        encoding="utf-8"
    )
    client = TestClient(app)

    ingest_response = client.post(
        "/ingest",
        json={"file_paths": [str(file_path)]}
    )
    ask_response = client.post(
        "/ask",
        json={"query": "How many leave days do contractors receive?", "top_k": 3}
    )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["indexed_documents"] == 1
    assert ask_response.status_code == 200
    assert "Contractors receive 10 days of leave." in ask_response.json()["answer"]

    # full pipeline: retriever -> reranker -> generator -> guardrails ->
    # API response, all the way through the real endpoint
    guardrail_flags = ask_response.json()["guardrail_flags"]
    assert guardrail_flags["pii_detected"] is False
    assert guardrail_flags["hallucination"] is False
    assert "groundedness" in guardrail_flags
