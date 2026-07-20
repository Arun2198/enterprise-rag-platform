import asyncio

from fastapi.testclient import TestClient

from app.main import app
from app.main import lifespan


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


def test_ask_accepts_optional_client_id(tmp_path):

    file_path = tmp_path / "policy.md"
    file_path.write_text("Some policy content for the client id test.", encoding="utf-8")
    client = TestClient(app)
    client.post("/ingest", json={"file_paths": [str(file_path)]})

    response = client.post(
        "/ask",
        json={"query": "What does the policy say?", "client_id": "requesting-user-1"}
    )

    assert response.status_code == 200


def test_admin_feature_flags_lists_the_reranker_flag():

    client = TestClient(app)

    response = client.get("/admin/feature-flags")

    assert response.status_code == 200
    names = [flag["name"] for flag in response.json()]
    assert "cross_encoder_reranker" in names


def test_admin_feature_flags_update_changes_rollout_percentage():

    client = TestClient(app)

    response = client.patch(
        "/admin/feature-flags/cross_encoder_reranker",
        json={"rollout_percentage": 25.0}
    )

    assert response.status_code == 200
    assert response.json()["rollout_percentage"] == 25.0

    # restore, so this test doesn't leak state into other tests sharing
    # the same module-level rag_service/platform_manager
    client.patch("/admin/feature-flags/cross_encoder_reranker", json={"rollout_percentage": 100.0})


def test_admin_feature_flags_update_unknown_flag_returns_404():

    client = TestClient(app)

    response = client.patch("/admin/feature-flags/does-not-exist", json={"enabled": True})

    assert response.status_code == 404


def test_admin_scheduler_lists_registered_jobs():

    client = TestClient(app)

    response = client.get("/admin/scheduler/jobs")

    assert response.status_code == 200
    job_ids = [job["job_id"] for job in response.json()]
    assert "backup" in job_ids
    assert "health_check" in job_ids


def test_admin_scheduler_trigger_runs_a_job_immediately():

    client = TestClient(app)

    response = client.post("/admin/scheduler/jobs/health_check/trigger")

    assert response.status_code == 200
    assert response.json()["job_id"] == "health_check"
    assert response.json()["success"] is True


def test_admin_scheduler_trigger_unknown_job_returns_404():

    client = TestClient(app)

    response = client.post("/admin/scheduler/jobs/does-not-exist/trigger")

    assert response.status_code == 404


def test_lifespan_starts_and_cleanly_cancels_the_scheduler_task():

    async def run():
        async with lifespan(app):
            pass

    asyncio.run(run())
