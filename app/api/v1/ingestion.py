"""Ingestion endpoints.

Registered at both `/ingest/document` (the path named in the master plan) and
`/ingestion/document` (the path the original implementation shipped), so the
documented contract and existing callers both work.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import require_ingestion_scope
from app.core.tenant_context import TenantContext
from app.models.payload import IngestionRequest, IngestionResponse
from app.services.ingestion_service import ingestion_service

router = APIRouter(tags=["Ingestion"])


async def _ingest(
    request: IngestionRequest, http_request: Request, ctx: TenantContext
) -> IngestionResponse:
    result = await ingestion_service.ingest_document(
        ctx=ctx,
        doc_id=request.doc_id,
        content=request.content,
        metadata=request.metadata,
    )
    return IngestionResponse(
        **result, request_id=getattr(http_request.state, "request_id", None)
    )


@router.post(
    "/ingest/document",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Chunk, embed, extract, resolve, and index a document into the tenant graph",
)
async def ingest_document(
    request: IngestionRequest,
    http_request: Request,
    ctx: TenantContext = Depends(require_ingestion_scope),
) -> IngestionResponse:
    return await _ingest(request, http_request, ctx)


@router.post(
    "/ingestion/document",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
    summary="Deprecated alias of /ingest/document",
)
async def ingest_document_legacy(
    request: IngestionRequest,
    http_request: Request,
    ctx: TenantContext = Depends(require_ingestion_scope),
) -> IngestionResponse:
    return await _ingest(request, http_request, ctx)
