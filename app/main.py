"""FastAPI application entry point for the Team B Graph RAG retrieval service."""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import (
    AuditLogMiddleware,
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from app.api.v1 import ingestion, retrieval, schema
from app.core.config import settings
from app.core.exceptions import (
    GraphRAGError,
    graph_rag_exception_handler,
    unhandled_exception_handler,
)
from app.models.payload import HealthResponse, ReadinessResponse
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.extraction_service import extraction_service
from app.services.reranker_service import reranker_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate configuration, warm the connection pool, and shut down cleanly."""
    problems = settings.validate_production()
    if problems:
        for problem in problems:
            logger.critical("PRODUCTION CONFIG ERROR: %s", problem)
        # Refuse to serve traffic with development secrets in production.
        raise RuntimeError(
            f"Refusing to start in production with {len(problems)} configuration "
            "violation(s). See logs above."
        )

    await arcadedb_client.start()
    ready = await arcadedb_client.is_ready()
    logger.info(
        "%s v%s starting | env=%s | arcadedb_ready=%s | tenants=%d",
        settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT,
        ready, len(settings.API_KEY_TENANT_MAP),
    )
    if not ready:
        logger.warning(
            "ArcadeDB is not reachable at %s. Requests will return 503 until it is.",
            settings.ARCADEDB_URL,
        )

    yield

    await arcadedb_client.close()
    logger.info("Shutdown complete.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Multi-tenant Graph RAG retrieval service over ArcadeDB. Database-per-tenant "
        "isolation, parameterized Cypher, hybrid vector + multi-hop graph retrieval "
        "with RRF fusion, and side-by-side latency/cost telemetry."
    ),
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Exception handlers: the whole typed taxonomy plus a catch-all that never leaks
# internals to the caller.
app.add_exception_handler(GraphRAGError, graph_rag_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Middleware. Starlette applies these in reverse registration order, so the last
# added runs outermost: RequestContext must be last so every other layer and every
# handler can read request_id.
app.add_middleware(AuditLogMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST"],
    allow_headers=[
        settings.API_KEY_HEADER,
        settings.TENANT_HEADER,
        settings.REQUEST_ID_HEADER,
        "Authorization",
        "Content-Type",
    ],
)
app.add_middleware(RequestContextMiddleware)

app.include_router(retrieval.router, prefix=settings.API_V1_STR)
app.include_router(ingestion.router, prefix=settings.API_V1_STR)
app.include_router(schema.router, prefix=settings.API_V1_STR)


@app.get("/health", response_model=HealthResponse, tags=["System Health"])
async def health_check() -> HealthResponse:
    """Liveness probe: is the process up? Never touches the database."""
    return HealthResponse(
        status="online",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["System Health"])
async def readiness_check() -> ReadinessResponse:
    """Readiness probe: can this instance actually serve retrieval requests?

    Split from /health so a load balancer removes an instance whose database is
    down instead of routing traffic that is guaranteed to fail.
    """
    db_ready = await arcadedb_client.is_ready()
    warnings: list[str] = []
    if not embedding_service.is_semantic:
        warnings.append(
            "Embedding model unavailable; using lexical fallback with reduced recall."
        )
    if not reranker_service.has_cross_encoder:
        warnings.append("Cross-encoder unavailable; using lexical reranking.")
    if extraction_service.active_backend == "regex":
        warnings.append("NER backend is regex; entity extraction quality is reduced.")
    if not settings.is_production:
        warnings.extend(settings.validate_production())

    return ReadinessResponse(
        ready=db_ready,
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        arcadedb_ready=db_ready,
        embedding_model=embedding_service.model_label,
        semantic_embeddings=embedding_service.is_semantic,
        cross_encoder_active=reranker_service.has_cross_encoder,
        extraction_backend=extraction_service.active_backend,
        configuration_warnings=warnings,
    )
