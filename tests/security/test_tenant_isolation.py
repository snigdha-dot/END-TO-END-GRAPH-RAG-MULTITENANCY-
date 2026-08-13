"""Security tests for Multi-Tenant Header validation and isolation."""
import pytest
from fastapi import HTTPException
from app.core.security import verify_tenant_header

@pytest.mark.asyncio
async def test_tenant_header_sanitization():
    valid_tenant = await verify_tenant_header("tech-support_123")
    assert valid_tenant == "tech-support_123"

@pytest.mark.asyncio
async def test_invalid_tenant_header_raises():
    with pytest.raises(HTTPException) as exc_info:
        await verify_tenant_header("!@#$%^&*()")
    assert exc_info.value.status_code == 400
