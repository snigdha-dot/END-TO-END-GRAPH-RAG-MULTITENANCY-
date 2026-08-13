"""Tenant schema inspection and provisioning: /api/v1/tenant/*.

Schema inspection is available to the tenant's own credential. Provisioning is an
administrative action gated on a separate admin key that is never issued to Team A —
otherwise any chatbot key could create databases at will.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status

from app.api.dependencies import authenticate_request, verify_admin_key
from app.core.tenant_context import TenantContext
from app.core.tenant_schema import schema_registry
from app.models.payload import TenantProvisionRequest, TenantSchemaResponse
from app.models.tenant import TenantProvisionResult
from app.services.arcadedb_client import arcadedb_client
from app.services.graph_schema_service import graph_schema_service

router = APIRouter(prefix="/tenant", tags=["Tenant Management"])


@router.get(
    "/schema",
    response_model=TenantSchemaResponse,
    summary="Inspect the approved graph schema for the authenticated tenant",
)
async def get_tenant_schema(
    ctx: TenantContext = Depends(authenticate_request),
) -> TenantSchemaResponse:
    schema = ctx.schema
    provisioned = await arcadedb_client.database_exists(schema.db_name)
    return TenantSchemaResponse(
        tenant_id=schema.tenant_id,
        db_name=schema.db_name,
        domain=schema.domain,
        display_name=schema.display_name,
        allowed_vertex_labels=sorted(schema.vertex_labels),
        allowed_edge_types=sorted(schema.edge_types),
        default_traversal_edges=schema.traversal_edges(),
        ner_labels=list(schema.ner_labels),
        provisioned=provisioned,
    )


@router.post(
    "/provision",
    response_model=TenantProvisionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Provision a tenant database, graph schema, and HNSW vector index (admin only)",
)
async def provision_tenant(
    request: TenantProvisionRequest,
    _admin: str = Depends(verify_admin_key),
) -> TenantProvisionResult:
    """Create the isolated database and full schema for a tenant. Idempotent."""
    created = await graph_schema_service.provision_tenant(request.tenant_id)
    return TenantProvisionResult(
        tenant_id=request.tenant_id,
        database=created["database"],
        vertex_types=created["vertex_types"],
        edge_types=created["edge_types"],
        indexes=created["indexes"],
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/schema/verify",
    summary="Verify that the declared schema exists in the tenant database (admin only)",
)
async def verify_tenant_schema(
    tenant_id: str,
    _admin: str = Depends(verify_admin_key),
) -> dict:
    return await graph_schema_service.verify_schema(tenant_id)


@router.get(
    "/registry",
    summary="List tenants with a curated domain schema (admin only)",
)
async def list_tenants(_admin: str = Depends(verify_admin_key)) -> dict:
    return {"tenants": schema_registry.all_tenants()}
