from app.schemas import AskRequest
from app.schemas import AskResponse
from app.schemas import IngestRequest
from app.schemas import IngestResponse
from app.service_factory import build_rag_service

try:
    from fastapi import FastAPI
except ImportError:  # pragma: no cover - keeps core tests runnable pre-API deps
    FastAPI = None


rag_service = build_rag_service()


if FastAPI is not None:
    app = FastAPI(
        title="Enterprise RAG Platform",
        version="0.1.0"
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        return rag_service.ingest(
            file_paths=request.file_paths,
            metadata=request.metadata
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        return rag_service.ask(
            query=request.query,
            top_k=request.top_k,
            metadata_filter=request.metadata_filter
        )
else:
    app = None
