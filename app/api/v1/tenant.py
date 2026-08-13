"""Tenant Management Endpoint: /api/v1/tenant."""
from datetime import datetime
from fastapi import APIRouter, Depends, status
from app.core.security import verify_api_key, verify_tenant_header
from app.models.tenant import TenantSchemaConfig
from app.services.arcadedb_client import arcadedb_client

router = APIRouter(prefix="/tenant", tags=["Tenant Management"])

@router.post(
    "/create",
    response_model=TenantSchemaConfig,
    status_code=status.HTTP_201_CREATED,
    summary="Provision a new isolated ArcadeDB multi-tenant database"
)
async def create_tenant_db(
    tenant_id: str,
    api_key: str = Depends(verify_api_key)
) -> TenantSchemaConfig:
    """Provision tenant database and initialize vertex/edge schemas."""
    db_name = await arcadedb_client.ensure_database_exists(tenant_id)
    return TenantSchemaConfig(
        tenant_id=tenant_id,
        db_name=db_name,
        created_at=datetime.utcnow().isoformat()
    )


@router.get(
    "/schema",
    response_model=TenantSchemaConfig,
    summary="Inspect tenant graph schema"
)
async def get_tenant_schema(
    x_tenant_id: str = Depends(verify_tenant_header),
    api_key: str = Depends(verify_api_key)
) -> TenantSchemaConfig:
    """Fetch target tenant allowed schemas and vertex classes."""
    return TenantSchemaConfig(
        tenant_id=x_tenant_id,
        db_name=f"tenant_{x_tenant_id.lower()}_kb",
        created_at=datetime.utcnow().isoformat()
    )
