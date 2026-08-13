"""Retrieval endpoint: POST /api/v1/retrieval/search."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import require_retrieval_scope
from app.core.tenant_context import TenantContext
from app.models.payload import RetrievalRequest, RetrievalResponse
from app.models.graph import Subgraph
from app.services.retrieval_service import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-hop hybrid graph retrieval with side-by-side cost and latency telemetry",
    responses={
        401: {"description": "Missing or invalid API key"},
        403: {"description": "Credential not authorized for the requested tenant"},
        404: {"description": "Tenant knowledge base is not provisioned"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Knowledge base temporarily unavailable"},
    },
)
async def search_retrieval(
    request: RetrievalRequest,
    http_request: Request,
    ctx: TenantContext = Depends(require_retrieval_scope),
) -> RetrievalResponse:
    """Shared retrieval API called by Team A chatbots.

    The tenant is taken from the authenticated credential, never from the payload.
    """
    result = await retrieval_service.execute_retrieval(
        ctx=ctx,
        query=request.user_query,
        max_depth=request.options.max_traversal_depth,
        top_k=request.options.top_k,
        include_vector_search=request.options.include_vector_search,
    )

    subgraph = result["subgraph"] if request.options.include_subgraph else Subgraph()

    return RetrievalResponse(
        tenant_id=ctx.tenant_id,
        query=request.user_query,
        subgraph=subgraph,
        context_passages=result["passages"],
        chunks=result["chunks"],
        linked_entities=result["linked_entities"],
        telemetry=result["telemetry"],
        request_id=getattr(http_request.state, "request_id", None),
    )
