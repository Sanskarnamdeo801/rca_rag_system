"""FastAPI entrypoint for the simplified RCA MVP."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, get_settings
from app.models import AnalyzeRequest, ErrorResponse, HealthResponse, Incident, IngestResponse, RCAResponse, SimilarIncident
from app.services.embedding_service import EmbeddingService
from app.services.ingest_service import IngestService
from app.services.rca_service import RCAService
from app.services.retrieval_service import RetrievalService
from app.services.vector_store import VectorStore
from app.utils.logger import configure_logging, get_logger

LOGGER = get_logger(__name__)


@dataclass
class AppServices:
    """Runtime service container."""

    settings: Settings
    embedding_service: EmbeddingService
    vector_store: VectorStore
    ingest_service: IngestService
    retrieval_service: RetrievalService
    rca_service: RCAService


def _build_services(settings: Settings) -> AppServices:
    embedding_service = EmbeddingService(settings)
    vector_store = VectorStore(settings, embedding_service)
    vector_store.initialize()

    def background_prepare() -> None:
        if settings.embedding_warmup:
            embedding_service.warmup()
        vector_store.warmup()

    threading.Thread(target=background_prepare, daemon=True).start()

    retrieval_service = RetrievalService(embedding_service, vector_store)
    ingest_service = IngestService(vector_store)
    rca_service = RCAService(settings, retrieval_service)
    return AppServices(
        settings=settings,
        embedding_service=embedding_service,
        vector_store=vector_store,
        ingest_service=ingest_service,
        retrieval_service=retrieval_service,
        rca_service=rca_service,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the MVP app."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = _build_services(resolved_settings)
        yield

    app = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=resolved_settings.static_dir), name="static")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=ErrorResponse(detail=str(exc.detail)).model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        LOGGER.error("Validation failed. error=%s", exc)
        return JSONResponse(status_code=422, content=ErrorResponse(detail="Invalid request payload.").model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled exception. error=%s", exc)
        return JSONResponse(status_code=500, content=ErrorResponse(detail="Internal server error.").model_dump())

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        return FileResponse(resolved_settings.static_dir / "index.html")

    @app.get("/health", response_model=HealthResponse, responses={500: {"model": ErrorResponse}})
    async def health() -> HealthResponse:
        services: AppServices = app.state.services
        return HealthResponse(
            status="ok",
            incidents_count=services.vector_store.size,
            index_ready=services.vector_store.index_ready,
            embedding_backend=services.embedding_service.backend_name,
            llm_provider=services.settings.llm_provider,
        )

    @app.post("/ingest", response_model=IngestResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    async def ingest(file: UploadFile = File(...)) -> IngestResponse:
        services: AppServices = app.state.services
        if not file.filename:
            raise HTTPException(status_code=400, detail="A file is required.")

        content = await file.read()
        if len(content) > services.settings.max_upload_size_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file exceeds the size limit.")

        try:
            added, duplicates, invalid_records = services.ingest_service.ingest(file.filename, content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return IngestResponse(
            status="success",
            logs_ingested=added,
            duplicates_skipped=duplicates,
            invalid_records=invalid_records,
            total_incidents=services.vector_store.size,
        )

    @app.post("/analyze", response_model=RCAResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    async def analyze(request: AnalyzeRequest) -> RCAResponse:
        services: AppServices = app.state.services
        return await services.rca_service.analyze(
            request.log,
            service_name=request.service_name,
            top_k=request.top_k,
        )

    @app.get("/incidents", response_model=list[Incident], responses={500: {"model": ErrorResponse}})
    async def incidents(service_name: str | None = Query(default=None, max_length=100)) -> list[Incident]:
        services: AppServices = app.state.services
        return services.vector_store.list_incidents(service_name)

    @app.get("/similar", response_model=list[SimilarIncident], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
    async def similar(
        message: str = Query(..., min_length=3, max_length=6000),
        service_name: str | None = Query(default=None, max_length=100),
        top_k: int = Query(default=5, ge=1, le=20),
    ) -> list[SimilarIncident]:
        services: AppServices = app.state.services
        return services.retrieval_service.find_similar(message, service_name=service_name, top_k=top_k)

    return app


app = create_app()
