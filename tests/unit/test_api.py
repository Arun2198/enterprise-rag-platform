import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.main import lifespan
from app.main import rag_service


def test_health_endpoint():

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_reports_503_when_startup_failed(monkeypatch):

    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]


def test_ingest_returns_503_when_rag_service_failed_to_initialize(monkeypatch):

    monkeypatch.setattr(main_module, "rag_service", None)
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.post("/ingest", json={"file_paths": ["sample_documents/AI-RMF-1stdraft.pdf"]})

    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]


def test_ask_returns_503_when_rag_service_failed_to_initialize(monkeypatch):

    monkeypatch.setattr(main_module, "rag_service", None)
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]


def test_admin_endpoints_report_the_real_failure_reason_not_a_generic_disabled_message(monkeypatch):

    monkeypatch.setattr(main_module, "platform_manager", None)
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.get("/admin/feature-flags")

    assert response.status_code == 404
    assert "model download failed" in response.json()["detail"]
    assert "MLOPS_ENABLED=false" not in response.json()["detail"]


def test_admin_endpoints_still_report_disabled_when_mlops_was_never_enabled(monkeypatch):

    monkeypatch.setattr(main_module, "platform_manager", None)
    monkeypatch.setattr(main_module, "startup_error", None)
    client = TestClient(app)

    response = client.get("/admin/feature-flags")

    assert response.status_code == 404
    assert "MLOPS_ENABLED=false" in response.json()["detail"]


def test_ingest_and_ask_endpoints(tmp_path, monkeypatch):

    # the live rag_service restricts /ingest to a configured directory
    # (see test_rag_service.py for the security tests) - point that
    # restriction at tmp_path for this test so it still exercises the
    # real, restricted code path rather than bypassing it
    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())

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


def test_ask_accepts_optional_client_id(tmp_path, monkeypatch):

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())

    file_path = tmp_path / "policy.md"
    file_path.write_text("Some policy content for the client id test.", encoding="utf-8")
    client = TestClient(app)
    client.post("/ingest", json={"file_paths": [str(file_path)]})

    response = client.post(
        "/ask",
        json={"query": "What does the policy say?", "client_id": "requesting-user-1"}
    )

    assert response.status_code == 200


def test_ingest_endpoint_rejects_a_path_outside_the_allowed_directory(tmp_path):

    outside_file = tmp_path / "secret.md"
    outside_file.write_text("# Secret\nShould never be readable via the API.", encoding="utf-8")
    client = TestClient(app)

    response = client.post("/ingest", json={"file_paths": [str(outside_file)]})

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_documents"] == 0
    assert "PATH_NOT_ALLOWED" in body["errors"][0]


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
