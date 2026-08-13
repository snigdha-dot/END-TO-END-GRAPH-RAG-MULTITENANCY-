"""Custom exception handlers for Team B API."""
from fastapi import Request, status
from fastapi.responses import JSONResponse

class DatabaseConnectionError(Exception):
    def __init__(self, detail: str):
        self.detail = detail

class TenantNotFoundError(Exception):
    def __init__(self, tenant_id: str):
        self.detail = f"Tenant database for '{tenant_id}' does not exist or is not initialized."

class EntityResolutionError(Exception):
    def __init__(self, detail: str):
        self.detail = detail


async def db_exception_handler(request: Request, exc: DatabaseConnectionError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"error": "Database Connection Failed", "detail": exc.detail}
    )

async def tenant_not_found_handler(request: Request, exc: TenantNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"error": "Tenant Not Found", "detail": exc.detail}
    )
