"""Retrieval API Endpoint: POST /api/v1/retrieval/search."""
from fastapi import APIRouter, Depends, Header, status
from app.core.security import verify_api_key, verify_tenant_header
from app.models.payload import RetrievalRequest, RetrievalResponse
from app.services.retrieval_service import retrieval_service

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])

@router.post(
    "/search",
    response_model=RetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-hop hybrid graph retrieval with side-by-side cost & latency telemetry"
)
async def search_retrieval(
    request: RetrievalRequest,
    api_key: str = Depends(verify_api_key),
    x_tenant_id: str = Depends(verify_tenant_header)
) -> RetrievalResponse:
    """Team B Shared Retrieval API called by Team A Chatbots."""
    # Ensure header tenant_id matches payload tenant_id
    tenant = x_tenant_id or request.tenant_id

    subgraph, passages, telemetry = await retrieval_service.execute_retrieval(
        tenant_id=tenant,
        query=request.user_query,
        max_depth=request.options.max_traversal_depth,
        top_k=request.options.top_k
    )

    return RetrievalResponse(
        tenant_id=tenant,
        query=request.user_query,
        subgraph=subgraph,
        context_passages=passages,
        telemetry=telemetry
    )
