import asyncio
import json
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


class _FakeTokenValidator:

    def __init__(self, user=None, error=None):
        self._user = user
        self._error = error

    def validate(self, token):
        if self._error:
            raise self._error

        return self._user


def _enable_auth(monkeypatch, validator):
    from dataclasses import replace

    monkeypatch.setattr(main_module, "settings", replace(main_module.settings, auth_enabled=True))
    monkeypatch.setattr(main_module, "token_validator", validator)


def test_ask_returns_401_with_no_authorization_header_when_auth_enabled(monkeypatch):

    _enable_auth(monkeypatch, _FakeTokenValidator())
    client = TestClient(app)

    response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 401


def test_ask_returns_401_with_a_malformed_authorization_header(monkeypatch):

    _enable_auth(monkeypatch, _FakeTokenValidator())
    client = TestClient(app)

    response = client.post("/ask", json={"query": "anything"}, headers={"Authorization": "NotBearer xyz"})

    assert response.status_code == 401


def test_ask_returns_401_when_the_token_fails_validation(monkeypatch):

    from app.auth import AuthenticationError

    _enable_auth(monkeypatch, _FakeTokenValidator(error=AuthenticationError("bad signature")))
    client = TestClient(app)

    response = client.post("/ask", json={"query": "anything"}, headers={"Authorization": "Bearer bad-token"})

    assert response.status_code == 401
    assert "bad signature" in response.json()["detail"]


def test_ask_succeeds_with_a_valid_token_that_has_query_permission(monkeypatch, tmp_path):

    from app.auth import AuthenticatedUser
    from mlops.schemas import Role

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())
    file_path = tmp_path / "policy.md"
    file_path.write_text("Some policy content for the auth test.", encoding="utf-8")
    TestClient(app).post("/ingest", json={"file_paths": [str(file_path)]})

    _enable_auth(monkeypatch, _FakeTokenValidator(
        user=AuthenticatedUser(subject="user-1", role=Role.READ_ONLY, claims={})
    ))
    client = TestClient(app)

    response = client.post("/ask", json={"query": "policy"}, headers={"Authorization": "Bearer good-token"})

    assert response.status_code == 200


def test_ask_debug_returns_403_for_a_read_only_role(monkeypatch, tmp_path):

    from app.auth import AuthenticatedUser
    from mlops.schemas import Role

    _enable_auth(monkeypatch, _FakeTokenValidator(
        user=AuthenticatedUser(subject="user-1", role=Role.READ_ONLY, claims={})
    ))
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"query": "anything"},
        headers={"Authorization": "Bearer good-token"}
    )

    assert response.status_code == 403


def test_ask_debug_succeeds_for_ml_engineer_and_matches_ask_shape(monkeypatch, tmp_path):

    from app.auth import AuthenticatedUser
    from mlops.schemas import Role

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())
    file_path = tmp_path / "policy.md"
    file_path.write_text("Contractors receive 10 days of leave per year.", encoding="utf-8")
    TestClient(app).post("/ingest", json={"file_paths": [str(file_path)]})

    _enable_auth(monkeypatch, _FakeTokenValidator(
        user=AuthenticatedUser(subject="user-1", role=Role.ML_ENGINEER, claims={})
    ))
    client = TestClient(app)

    response = client.post(
        "/ask/debug",
        json={"query": "How many leave days do contractors receive?", "top_k": 3},
        headers={"Authorization": "Bearer good-token"}
    )

    assert response.status_code == 200
    body = response.json()
    assert "Contractors receive 10 days of leave" in body["response"]["answer"]
    trace = body["trace"]
    assert trace["query"] == "How many leave days do contractors receive?"
    assert len(trace["dense_candidates"]) >= 1
    assert len(trace["bm25_candidates"]) >= 1
    assert "embedding" in trace["stage_timings_ms"]
    assert "total" in trace["stage_timings_ms"]


def test_ask_debug_returns_503_when_rag_service_failed_to_initialize(monkeypatch):

    monkeypatch.setattr(main_module, "rag_service", None)
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.post("/ask/debug", json={"query": "anything"})

    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]


def test_ingest_returns_403_for_a_read_only_role(monkeypatch, tmp_path):

    from app.auth import AuthenticatedUser
    from mlops.schemas import Role

    _enable_auth(monkeypatch, _FakeTokenValidator(
        user=AuthenticatedUser(subject="user-1", role=Role.READ_ONLY, claims={})
    ))
    client = TestClient(app)

    response = client.post(
        "/ingest",
        json={"file_paths": [str(tmp_path / "x.md")]},
        headers={"Authorization": "Bearer good-token"}
    )

    assert response.status_code == 403


def test_ingest_succeeds_for_a_data_scientist_role(monkeypatch, tmp_path):

    from app.auth import AuthenticatedUser
    from mlops.schemas import Role

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())
    file_path = tmp_path / "policy.md"
    file_path.write_text("Some policy content.", encoding="utf-8")

    _enable_auth(monkeypatch, _FakeTokenValidator(
        user=AuthenticatedUser(subject="user-1", role=Role.DATA_SCIENTIST, claims={})
    ))
    client = TestClient(app)

    response = client.post(
        "/ingest",
        json={"file_paths": [str(file_path)]},
        headers={"Authorization": "Bearer good-token"}
    )

    assert response.status_code == 200


def test_requests_still_succeed_without_a_token_when_auth_is_disabled():

    client = TestClient(app)

    response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 200


def test_ready_endpoint_returns_ready_when_healthy():

    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_rag_service_failed_to_initialize(monkeypatch):

    monkeypatch.setattr(main_module, "rag_service", None)
    monkeypatch.setattr(main_module, "startup_error", "RuntimeError: model download failed")
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "model download failed" in response.json()["detail"]


def test_ready_returns_503_when_the_vector_store_health_check_fails(monkeypatch):

    class UnhealthyVectorStore:
        def health_check(self):
            raise ConnectionError("cluster unreachable")

    monkeypatch.setattr(rag_service, "vector_store", UnhealthyVectorStore())
    client = TestClient(app)

    response = client.get("/ready")

    assert response.status_code == 503
    assert "cluster unreachable" in response.json()["detail"]


def test_ready_does_not_require_a_health_check_method():
    """
    InMemoryVectorStore has no health_check() at all - readiness must not
    assume every vector store implementation has one.
    """
    client = TestClient(app)

    assert not hasattr(rag_service.vector_store, "health_check")
    response = client.get("/ready")

    assert response.status_code == 200


def test_ask_rejects_a_query_over_the_max_length():

    client = TestClient(app)

    response = client.post("/ask", json={"query": "x" * 2001})

    assert response.status_code == 422


def test_ask_accepts_a_query_at_the_max_length():

    client = TestClient(app)

    response = client.post("/ask", json={"query": "x" * 2000})

    assert response.status_code == 200


def test_ingest_rejects_more_than_the_max_file_paths_per_request():

    client = TestClient(app)

    response = client.post("/ingest", json={"file_paths": [f"file_{i}.md" for i in range(51)]})

    assert response.status_code == 422


def test_unhandled_exception_returns_a_sanitized_500_not_the_raw_error(monkeypatch):

    def _boom(*args, **kwargs):
        raise RuntimeError("super secret internal detail: db_password=hunter2")

    monkeypatch.setattr(rag_service, "ask", _boom)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/ask", json={"query": "anything"})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "hunter2" not in response.text
    assert "db_password" not in response.text


def test_delete_document_endpoint_removes_chunks(tmp_path, monkeypatch):

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())
    file_path = tmp_path / "todelete.md"
    file_path.write_text("Some content to delete later.", encoding="utf-8")
    client = TestClient(app)
    client.post("/ingest", json={"file_paths": [str(file_path)]})

    response = client.delete("/documents/todelete")

    assert response.status_code == 200
    assert response.json()["document_id"] == "todelete"
    assert response.json()["deleted_chunks"] >= 1


def test_delete_document_endpoint_returns_zero_for_unknown_document():

    client = TestClient(app)

    response = client.delete("/documents/never-existed")

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 0


def test_reindex_endpoint_replaces_document(tmp_path, monkeypatch):

    monkeypatch.setattr(rag_service, "ingest_allowed_dir", Path(tmp_path).resolve())
    file_path = tmp_path / "reindexme.md"
    file_path.write_text("Original content here for reindexing test.", encoding="utf-8")
    client = TestClient(app)
    client.post("/ingest", json={"file_paths": [str(file_path)]})

    file_path.write_text("Updated content here for reindexing test.", encoding="utf-8")
    response = client.post("/documents/reindex", json={"file_path": str(file_path)})

    assert response.status_code == 200
    assert response.json()["indexed_documents"] == 1


def test_upload_document_returns_503_when_async_ingestion_not_configured():

    client = TestClient(app)

    response = client.post("/documents", files={"file": ("test.md", b"some content", "text/markdown")})

    assert response.status_code == 503


def test_get_job_status_returns_503_when_not_configured():

    client = TestClient(app)

    response = client.get("/documents/jobs/some-job-id")

    assert response.status_code == 503


class _FakeS3StoreForUpload:

    def __init__(self):
        self.bucket_name = "fake-bucket"
        self.uploaded = []

    def upload(self, local_path, document_id, original_filename=None):
        self.uploaded.append({"local_path": local_path, "document_id": document_id, "filename": original_filename})
        return f"raw/{document_id}.md"


class _FakeJobStoreForUpload:

    def __init__(self):
        self.jobs = {}

    def create_job(self, job_id, document_id, s3_key):
        record = {
            "job_id": job_id, "document_id": document_id, "s3_key": s3_key,
            "status": "RECEIVED", "error": None,
            "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00"
        }
        self.jobs[job_id] = record
        return record

    def get_job(self, job_id):
        return self.jobs.get(job_id)


def test_upload_document_uploads_to_s3_and_creates_a_job(monkeypatch):

    fake_s3 = _FakeS3StoreForUpload()
    fake_jobs = _FakeJobStoreForUpload()
    monkeypatch.setattr(main_module, "s3_document_store", fake_s3)
    monkeypatch.setattr(main_module, "ingestion_job_store", fake_jobs)
    monkeypatch.setattr(main_module, "sqs_client", None)
    client = TestClient(app)

    response = client.post("/documents", files={"file": ("policy.md", b"some content", "text/markdown")})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "RECEIVED"
    assert body["job_id"] in fake_jobs.jobs
    assert fake_s3.uploaded[0]["filename"] == "policy.md"


def test_upload_document_enqueues_an_sqs_message_when_configured(monkeypatch):

    fake_s3 = _FakeS3StoreForUpload()
    fake_jobs = _FakeJobStoreForUpload()

    class FakeSQS:
        def __init__(self):
            self.sent = []
        def send_message(self, QueueUrl, MessageBody):
            self.sent.append({"QueueUrl": QueueUrl, "MessageBody": MessageBody})

    fake_sqs = FakeSQS()
    monkeypatch.setattr(main_module, "s3_document_store", fake_s3)
    monkeypatch.setattr(main_module, "ingestion_job_store", fake_jobs)
    monkeypatch.setattr(main_module, "sqs_client", fake_sqs)
    monkeypatch.setattr(main_module, "settings", replace_setting(main_module.settings, "sqs_queue_url", "https://sqs.example/queue"))
    client = TestClient(app)

    response = client.post("/documents", files={"file": ("policy.md", b"some content", "text/markdown")})

    assert response.status_code == 200
    assert len(fake_sqs.sent) == 1
    body = json.loads(fake_sqs.sent[0]["MessageBody"])
    assert body["document_id"] == response.json()["document_id"]


def test_upload_document_rejects_a_disallowed_file_type(monkeypatch):

    from ingestion.s3_document_store import S3ValidationError

    class RejectingS3Store:
        bucket_name = "fake-bucket"
        def upload(self, local_path, document_id, original_filename=None):
            raise S3ValidationError("file type not allowed")

    monkeypatch.setattr(main_module, "s3_document_store", RejectingS3Store())
    monkeypatch.setattr(main_module, "ingestion_job_store", _FakeJobStoreForUpload())
    client = TestClient(app)

    response = client.post("/documents", files={"file": ("malware.exe", b"x", "application/octet-stream")})

    assert response.status_code == 422


def test_get_job_status_returns_the_job_record(monkeypatch):

    fake_jobs = _FakeJobStoreForUpload()
    fake_jobs.create_job("job-1", "doc-1", "raw/doc-1.md")
    monkeypatch.setattr(main_module, "ingestion_job_store", fake_jobs)
    client = TestClient(app)

    response = client.get("/documents/jobs/job-1")

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
    assert response.json()["status"] == "RECEIVED"


def test_get_job_status_returns_404_for_an_unknown_job(monkeypatch):

    monkeypatch.setattr(main_module, "ingestion_job_store", _FakeJobStoreForUpload())
    client = TestClient(app)

    response = client.get("/documents/jobs/does-not-exist")

    assert response.status_code == 404


def replace_setting(settings, field_name, value):
    from dataclasses import replace
    return replace(settings, **{field_name: value})
