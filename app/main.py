"""Main FastAPI Application Entry Point for Team B Graph RAG Service."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError, TenantNotFoundError,
    db_exception_handler, tenant_not_found_handler
)
from app.services.arcadedb_client import arcadedb_client
from app.api.v1 import retrieval, ingestion, tenant

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Start HTTP client pool & verify ArcadeDB readiness."""
    await arcadedb_client.start()
    ready = await arcadedb_client.is_ready()
    print(f"[{settings.PROJECT_NAME}] ArcadeDB Ready Status: {ready}")
    yield
    await arcadedb_client.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Team B Multi-Tenant Graph RAG Retrieval Service with ArcadeDB, Cypher Parameterization, and Side-by-Side Cost Telemetry.",
    lifespan=lifespan,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Exception Handlers
app.add_exception_handler(DatabaseConnectionError, db_exception_handler)
app.add_exception_handler(TenantNotFoundError, tenant_not_found_handler)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Router Endpoints
app.include_router(retrieval.router, prefix=settings.API_V1_STR)
app.include_router(ingestion.router, prefix=settings.API_V1_STR)
app.include_router(tenant.router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["System Health"])
async def health_check():
    db_ready = await arcadedb_client.is_ready()
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "arcadedb_ready": db_ready
    }
