import asyncio
import contextlib
import logging

from app.config import load_settings
from app.schemas import AskRequest
from app.schemas import AskResponse
from app.schemas import FeatureFlagResponse
from app.schemas import FeatureFlagUpdateRequest
from app.schemas import IngestRequest
from app.schemas import IngestResponse
from app.schemas import JobRunResponse
from app.schemas import ScheduledJobResponse
from app.service_factory import build_platform_manager
from app.service_factory import build_rag_service
from mlops.feature_flags import FlagNotFoundError
from mlops.scheduler import JobNotFoundError

try:
    from fastapi import FastAPI
    from fastapi import HTTPException
except ImportError:  # pragma: no cover - keeps core tests runnable pre-API deps
    FastAPI = None
    HTTPException = None


logger = logging.getLogger(__name__)

settings = load_settings()
platform_manager = build_platform_manager(settings)
rag_service = build_rag_service(settings, platform_manager=platform_manager)

if platform_manager is not None and settings.scheduler_enabled:
    platform_manager.scheduler.register(
        job_id="health_check",
        name="Index health check",
        interval_seconds=settings.scheduler_interval_seconds,
        callable_=lambda: logger.info(
            "scheduled_health_check",
            extra={"indexed_chunks": len(rag_service.vector_store)}
        )
    )


async def _scheduler_loop() -> None:
    """
    Scheduler owns no thread/loop of its own by design (see
    mlops.scheduler.Scheduler) - this is the "whatever actually owns
    scheduling in a deployment" piece for the FastAPI app specifically.
    """
    while True:
        await asyncio.sleep(settings.scheduler_interval_seconds)
        platform_manager.scheduler.run_due_jobs()


if FastAPI is not None:
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        scheduler_task = None

        if platform_manager is not None and settings.scheduler_enabled:
            scheduler_task = asyncio.create_task(_scheduler_loop())

        yield

        if scheduler_task is not None:
            scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scheduler_task

    app = FastAPI(
        title="Enterprise RAG Platform",
        version="0.1.0",
        lifespan=lifespan
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        return rag_service.ingest(request.file_paths)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return rag_service.ask(
            query=request.query,
            top_k=request.top_k,
            client_id=request.client_id
        )

    @app.get("/admin/feature-flags", response_model=list[FeatureFlagResponse])
    def list_feature_flags() -> list[FeatureFlagResponse]:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail="MLOPS_ENABLED=false - no feature flags")

        return [FeatureFlagResponse(**vars(flag)) for flag in platform_manager.feature_flags.list()]

    @app.patch("/admin/feature-flags/{name}", response_model=FeatureFlagResponse)
    def update_feature_flag(name: str, request: FeatureFlagUpdateRequest) -> FeatureFlagResponse:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail="MLOPS_ENABLED=false - no feature flags")

        try:
            flag = platform_manager.feature_flags.get(name)

            if request.enabled is not None:
                flag = platform_manager.feature_flags.set_enabled(name, request.enabled)

            if request.rollout_percentage is not None:
                flag = platform_manager.feature_flags.set_rollout_percentage(
                    name, request.rollout_percentage
                )
        except FlagNotFoundError:
            raise HTTPException(status_code=404, detail=f"unknown feature flag: {name}")

        return FeatureFlagResponse(**vars(flag))

    @app.get("/admin/scheduler/jobs", response_model=list[ScheduledJobResponse])
    def list_scheduled_jobs() -> list[ScheduledJobResponse]:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail="MLOPS_ENABLED=false - no scheduler")

        return [ScheduledJobResponse(**vars(job)) for job in platform_manager.scheduler.list_jobs()]

    @app.post("/admin/scheduler/jobs/{job_id}/trigger", response_model=JobRunResponse)
    def trigger_scheduled_job(job_id: str) -> JobRunResponse:
        if platform_manager is None:
            raise HTTPException(status_code=404, detail="MLOPS_ENABLED=false - no scheduler")

        try:
            run = platform_manager.scheduler.trigger(job_id)
        except JobNotFoundError:
            raise HTTPException(status_code=404, detail=f"unknown scheduled job: {job_id}")

        return JobRunResponse(**vars(run))
else:
    app = None
