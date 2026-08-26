import asyncio
import contextlib
import json
import logging
import os
import tempfile
import uuid

from app.auth import AuthenticatedUser
from app.auth import AuthenticationError
from app.auth import OIDCTokenValidator
from app.config import load_settings
from app.schemas import AskDebugResponse
from app.schemas import AskRequest
from app.schemas import AskResponse
from app.schemas import CandidateTraceResponse
from app.schemas import DocumentDeleteResponse
from app.schemas import DocumentUploadResponse
from app.schemas import FeatureFlagResponse
from app.schemas import FeatureFlagUpdateRequest
from app.schemas import IngestRequest
from app.schemas import IngestResponse
from app.schemas import JobRunResponse
from app.schemas import JobStatusResponse
from app.schemas import ReindexRequest
from app.schemas import RetrievalTraceResponse
from app.schemas import ScheduledJobResponse
from app.service_factory import build_ingestion_job_store
from app.service_factory import build_platform_manager
from app.service_factory import build_rag_service
from app.service_factory import build_s3_document_store
from app.service_factory import build_sqs_client
from app.service_factory import build_sqs_ingestion_worker
from ingestion.s3_document_store import S3ValidationError
from mlops.feature_flags import FlagNotFoundError
from mlops.ingestion_job_store import JobStatus
from mlops.permissions import PermissionDeniedError
from mlops.permissions import require_permission as mlops_require_permission
from mlops.scheduler import JobNotFoundError
from mlops.schemas import Permission
from mlops.schemas import Role

try:
    from fastapi import Depends
    from fastapi import FastAPI
    from fastapi import File
    from fastapi import Header
    from fastapi import HTTPException
    from fastapi import UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - keeps core tests runnable pre-API deps
    CORSMiddleware = None  # type: ignore[assignment,misc]
    Depends = None  # type: ignore[assignment]
    FastAPI = None  # type: ignore[assignment,misc]
    File = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    UploadFile = None  # type: ignore[assignment,misc]

# Full-access synthetic identity used when AUTH_ENABLED=false (the
# default) - preserves today's fully-open behavior exactly, rather than
# every route needing an "is auth even on" branch. Real deployments turn
# auth on by setting AUTH_ENABLED=true plus the three OIDC_* variables.
_AUTH_DISABLED_USER = AuthenticatedUser(subject="auth-disabled", role=Role.ADMINISTRATOR, claims={})


logger = logging.getLogger(__name__)

settings = load_settings()

# Building the RAG pipeline means loading real ML models (embedder,
# reranker) - a download hiccup, HuggingFace outage, or OOM here must not
# take the whole ASGI process down with it. An unguarded failure here
# used to mean the app never finishes importing, so not even /health was
# reachable - there was no way to tell "the app is up but a dependency
# failed to load" from "the app is completely dead". Guarding it means
# the process always starts and stays diagnosable; /health reports the
# degraded state explicitly instead.
platform_manager = None
rag_service = None
token_validator: OIDCTokenValidator | None = None
s3_document_store = None
ingestion_job_store = None
sqs_client = None
sqs_ingestion_worker = None
startup_error: str | None = None

try:
    platform_manager = build_platform_manager(settings)
    rag_service = build_rag_service(settings, platform_manager=platform_manager)
    s3_document_store = build_s3_document_store(settings)
    ingestion_job_store = build_ingestion_job_store(settings)
    sqs_client = build_sqs_client(settings)

    if s3_document_store is not None and ingestion_job_store is not None:
        sqs_ingestion_worker = build_sqs_ingestion_worker(
            settings, rag_service, s3_document_store, ingestion_job_store, sqs_client
        )

    if settings.auth_enabled:
        if not (settings.oidc_issuer and settings.oidc_audience and settings.oidc_jwks_url):
            raise ValueError(
                "AUTH_ENABLED=true requires OIDC_ISSUER, OIDC_AUDIENCE, "
                "and OIDC_JWKS_URL to all be set."
            )

        token_validator = OIDCTokenValidator(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_url=settings.oidc_jwks_url,
            role_claim=settings.oidc_role_claim
        )
except Exception as ex:
    startup_error = f"{type(ex).__name__}: {ex}"
    logger.error("rag_service_initialization_failed", extra={"error": startup_error})

if platform_manager is not None and rag_service is not None and settings.scheduler_enabled:
    # Bind to a local so the lambda closes over a name mypy (and future
    # readers) can see is non-None, rather than the reassignable module
    # global - rag_service/platform_manager are only ever set once above,
    # but that invariant isn't visible across the closure boundary.
    _rag_service_for_health_check = rag_service
    platform_manager.scheduler.register(
        job_id="health_check",
        name="Index health check",
        interval_seconds=settings.scheduler_interval_seconds,
        callable_=lambda: logger.info(
            "scheduled_health_check",
            extra={"indexed_chunks": _rag_service_for_health_check.vector_store.count()}
        )
    )


async def _scheduler_loop() -> None:
    """
    Scheduler owns no thread/loop of its own by design (see
    mlops.scheduler.Scheduler) - this is the "whatever actually owns
    scheduling in a deployment" piece for the FastAPI app specifically.
    """
    # Only ever scheduled as an asyncio task from lifespan() when
    # platform_manager is not None (see below) - the assert documents
    # that call-site invariant for mypy, which can't see across it.
    assert platform_manager is not None
    while True:
        await asyncio.sleep(settings.scheduler_interval_seconds)
        platform_manager.scheduler.run_due_jobs()


async def _sqs_ingestion_loop() -> None:
    """
    SQSIngestionWorker owns no polling loop of its own either, same
    caller-drives-the-clock philosophy as the scheduler. Running it as an
    in-process asyncio task (rather than a separate worker service/task
    definition) is a deliberate scale-appropriate choice, not a
    corner-cut placeholder - a low-volume ingestion queue doesn't justify
    a second ECS service yet; this is the same pattern the scheduler
    already uses for exactly this reason. Moving to a dedicated worker
    task later needs zero changes to SQSIngestionWorker itself, only to
    what calls poll_once().
    """
    # Only ever scheduled as an asyncio task from lifespan() when
    # sqs_ingestion_worker is not None (see below).
    assert sqs_ingestion_worker is not None
    while True:
        await asyncio.sleep(settings.sqs_poll_interval_seconds)
        try:
            sqs_ingestion_worker.poll_once()
        except Exception as ex:
            logger.warning("sqs_poll_failed", extra={"error": f"{type(ex).__name__}: {ex}"})


if FastAPI is not None:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler_task = None
        sqs_task = None

        if platform_manager is not None and settings.scheduler_enabled:
            scheduler_task = asyncio.create_task(_scheduler_loop())

        if sqs_ingestion_worker is not None:
            sqs_task = asyncio.create_task(_sqs_ingestion_loop())

        yield

        for task in (scheduler_task, sqs_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(
        title="Enterprise RAG Platform",
        version="0.1.0",
        lifespan=lifespan
    )

    if settings.cors_allowed_origins:
        # Off by default (no origins configured = no CORSMiddleware at
        # all, unchanged from before this existed) - a browser-hosted
        # frontend on a different origin needs this explicitly opted in
        # via CORS_ALLOWED_ORIGINS, since this API sits behind real
        # bearer-token auth rather than relying on browser same-origin
        # policy for protection.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc: Exception):
        """
        Defense in depth against leaking internals on a genuinely
        unexpected exception (Starlette's own default already avoids
        echoing a traceback when debug=False, which this app never sets,
        but that's implicit; this makes it explicit and guarantees the
        same sanitized shape as every other error response). Full detail
        goes to the server log where an operator can see it; the response
        never contains the exception message, a stack trace, or anything
        that might be an internal path/credential/secret.
        """
        logger.error(
            "unhandled_exception",
            extra={"path": str(request.url.path), "error": f"{type(exc).__name__}: {exc}"},
            exc_info=exc
        )
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    def get_current_user(
        authorization: str | None = Header(default=None)
    ) -> AuthenticatedUser:
        if not settings.auth_enabled:
            return _AUTH_DISABLED_USER

        if token_validator is None:
            # auth_enabled=True but startup failed to build the validator
            # (e.g. malformed JWKS URL) - same 503-with-real-detail pattern
            # as the other service-unavailable checks below, rather than
            # an unguarded AttributeError surfacing as a generic 500.
            raise HTTPException(
                status_code=503,
                detail=f"auth service unavailable: {startup_error}"
            )

        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing or malformed Authorization header")

        token = authorization.split(" ", 1)[1].strip()

        try:
            return token_validator.validate(token)
        except AuthenticationError as ex:
            raise HTTPException(status_code=401, detail=f"invalid token: {ex}") from ex

    def require_permission(permission: Permission):
        """
        Dependency factory - Depends(require_permission(Permission.QUERY))
        authenticates the caller (or passes through the full-access
        synthetic user when auth is disabled) and then enforces RBAC via
        the existing mlops.permissions logic, so authorization actually
        affects execution rather than just being defined in Python.
        """
        def _dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
            try:
                mlops_require_permission(user.role, permission)
            except PermissionDeniedError as ex:
                raise HTTPException(status_code=403, detail=str(ex)) from ex

            return user

        return _dependency

    @app.get("/health")
    def health() -> dict[str, str]:
        """
        Liveness only - is the process up at all. Deliberately does not
        touch the vector store, embedder, or any external dependency;
        that's what /ready is for.
        """
        if startup_error is not None:
            raise HTTPException(
                status_code=503,
                detail=f"service failed to initialize: {startup_error}"
            )

        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        """
        Readiness - can this instance actually serve traffic right now.
        Checks only the one critical, cheap-to-check dependency (the
        vector store's own health, when the store exposes one) - never
        the LLM: an LLM call is slow, costs money per invocation, and
        answers a different question ("is generation currently working")
        than readiness needs ("should the load balancer send this
        instance traffic").
        """
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        health_check = getattr(rag_service.vector_store, "health_check", None)

        if health_check is not None:
            try:
                health_check()
            except Exception as ex:
                raise HTTPException(
                    status_code=503,
                    detail=f"vector store unreachable: {type(ex).__name__}: {ex}"
                ) from ex

        return {"status": "ready"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(
        request: IngestRequest,
        user: AuthenticatedUser = Depends(require_permission(Permission.UPLOAD_DOCUMENT))
    ) -> IngestResponse:
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        return rag_service.ingest(request.file_paths)

    @app.delete("/documents/{document_id}", response_model=DocumentDeleteResponse)
    def delete_document(
        document_id: str,
        user: AuthenticatedUser = Depends(require_permission(Permission.DELETE_DOCUMENT))
    ) -> DocumentDeleteResponse:
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        deleted_count = rag_service.delete_document(document_id)
        return DocumentDeleteResponse(document_id=document_id, deleted_chunks=deleted_count)

    @app.post("/documents/reindex", response_model=IngestResponse)
    def reindex_document(
        request: ReindexRequest,
        user: AuthenticatedUser = Depends(require_permission(Permission.UPLOAD_DOCUMENT))
    ) -> IngestResponse:
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        return rag_service.reindex_document(request.file_path)

    @app.post("/documents", response_model=DocumentUploadResponse)
    async def upload_document(
        file: UploadFile = File(...),
        user: AuthenticatedUser = Depends(require_permission(Permission.UPLOAD_DOCUMENT))
    ) -> DocumentUploadResponse:
        """
        Asynchronous ingestion entry point: uploads to S3, enqueues one
        SQS message, and returns immediately with RECEIVED status rather
        than blocking the request on parse/chunk/embed/index - see
        GET /documents/jobs/{job_id} to poll for completion. Requires
        S3_BUCKET + ASYNC_INGESTION_ENABLED + SQS_QUEUE_URL to all be
        configured; the synchronous /ingest endpoint remains available
        with zero S3/SQS configuration either way.
        """
        if s3_document_store is None or ingestion_job_store is None:
            raise HTTPException(
                status_code=503,
                detail="asynchronous ingestion is not configured (S3_BUCKET not set)"
            )

        document_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        suffix = os.path.splitext(file.filename or "")[1]
        fd, temp_path = tempfile.mkstemp(suffix=suffix)

        try:
            with os.fdopen(fd, "wb") as f:
                f.write(await file.read())

            key = s3_document_store.upload(temp_path, document_id, original_filename=file.filename)
        except S3ValidationError as ex:
            raise HTTPException(status_code=422, detail=str(ex)) from ex
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        ingestion_job_store.create_job(job_id, document_id, key)

        if sqs_client is not None:
            sqs_client.send_message(
                QueueUrl=settings.sqs_queue_url,
                MessageBody=json.dumps({"job_id": job_id, "document_id": document_id, "key": key})
            )

        return DocumentUploadResponse(document_id=document_id, job_id=job_id, status=JobStatus.RECEIVED.value)

    @app.get("/documents/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(
        job_id: str,
        user: AuthenticatedUser = Depends(require_permission(Permission.QUERY))
    ) -> JobStatusResponse:
        if ingestion_job_store is None:
            raise HTTPException(
                status_code=503,
                detail="asynchronous ingestion is not configured (S3_BUCKET not set)"
            )

        record = ingestion_job_store.get_job(job_id)

        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")

        return JobStatusResponse(**record)

    @app.post("/ask", response_model=AskResponse)
    def ask(
        request: AskRequest,
        user: AuthenticatedUser = Depends(require_permission(Permission.QUERY))
    ) -> AskResponse:
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        return rag_service.ask(
            query=request.query,
            top_k=request.top_k,
            client_id=request.client_id,
            # from the validated token's own claims, never the request
            # body - a caller cannot claim arbitrary access groups for
            # themselves.
            access_groups=user.claims.get("access_groups")
        )

    @app.post("/ask/debug", response_model=AskDebugResponse)
    def ask_debug(
        request: AskRequest,
        user: AuthenticatedUser = Depends(require_permission(Permission.DEBUG_QUERY))
    ) -> AskDebugResponse:
        """
        Same as /ask, but also returns the full per-stage retrieval trace
        (embedding/dense/BM25/RRF/rerank/generation/groundedness/guardrail
        detail and latency) - gated behind DEBUG_QUERY since it exposes
        internal scoring detail (raw chunk ids, per-stage scores) that
        regular QUERY-only callers shouldn't see.
        """
        if rag_service is None:
            raise HTTPException(
                status_code=503,
                detail=f"service unavailable: {startup_error}"
            )

        response, trace = rag_service.ask_with_trace(
            query=request.query,
            top_k=request.top_k,
            client_id=request.client_id,
            access_groups=user.claims.get("access_groups")
        )
        return AskDebugResponse(
            response=response,
            trace=RetrievalTraceResponse(
                query=trace.query,
                embedding_provider=trace.embedding_provider,
                embedding_dimensions=trace.embedding_dimensions,
                dense_candidates=[
                    CandidateTraceResponse(**vars(c)) for c in trace.dense_candidates
                ],
                bm25_candidates=[
                    CandidateTraceResponse(**vars(c)) for c in trace.bm25_candidates
                ],
                fused_candidates=[
                    CandidateTraceResponse(**vars(c)) for c in trace.fused_candidates
                ],
                reranker_used=trace.reranker_used,
                reranked_candidates=[
                    CandidateTraceResponse(**vars(c)) for c in trace.reranked_candidates
                ],
                final_chunk_ids=trace.final_chunk_ids,
                generation_provider=trace.generation_provider,
                groundedness=trace.groundedness,
                guardrail_findings=trace.guardrail_findings,
                stage_timings_ms=trace.stage_timings_ms
            )
        )

    @app.get("/admin/feature-flags", response_model=list[FeatureFlagResponse])
    def list_feature_flags(
        user: AuthenticatedUser = Depends(require_permission(Permission.VIEW))
    ) -> list[FeatureFlagResponse]:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail=_mlops_unavailable_detail())

        return [FeatureFlagResponse(**vars(flag)) for flag in platform_manager.feature_flags.list()]

    @app.patch("/admin/feature-flags/{name}", response_model=FeatureFlagResponse)
    def update_feature_flag(
        name: str,
        request: FeatureFlagUpdateRequest,
        user: AuthenticatedUser = Depends(require_permission(Permission.TOGGLE_FEATURE_FLAG))
    ) -> FeatureFlagResponse:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail=_mlops_unavailable_detail())

        try:
            flag = platform_manager.feature_flags.get(name)

            if request.enabled is not None:
                flag = platform_manager.feature_flags.set_enabled(name, request.enabled)

            if request.rollout_percentage is not None:
                flag = platform_manager.feature_flags.set_rollout_percentage(
                    name, request.rollout_percentage
                )
        except FlagNotFoundError as ex:
            raise HTTPException(status_code=404, detail=f"unknown feature flag: {name}") from ex

        return FeatureFlagResponse(**vars(flag))

    @app.get("/admin/scheduler/jobs", response_model=list[ScheduledJobResponse])
    def list_scheduled_jobs(
        user: AuthenticatedUser = Depends(require_permission(Permission.VIEW))
    ) -> list[ScheduledJobResponse]:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail=_mlops_unavailable_detail())

        return [ScheduledJobResponse(**vars(job)) for job in platform_manager.scheduler.list_jobs()]

    @app.post("/admin/scheduler/jobs/{job_id}/trigger", response_model=JobRunResponse)
    def trigger_scheduled_job(
        job_id: str,
        user: AuthenticatedUser = Depends(require_permission(Permission.TRIGGER_DEPLOYMENT))
    ) -> JobRunResponse:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail=_mlops_unavailable_detail())

        try:
            run = platform_manager.scheduler.trigger(job_id)
        except JobNotFoundError as ex:
            raise HTTPException(status_code=404, detail=f"unknown scheduled job: {job_id}") from ex

        return JobRunResponse(**vars(run))

    def _mlops_unavailable_detail() -> str:
        if startup_error is not None:
            return f"mlops unavailable: {startup_error}"

        return "MLOPS_ENABLED=false - no feature flags or scheduler"
else:
    app = None  # type: ignore[assignment]
