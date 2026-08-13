"""Custom exception taxonomy and handlers for Team B API.

Design rule: a failure must never be indistinguishable from an empty result.
Every database or model failure raises; only a genuinely empty result set
returns an empty list.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class GraphRAGError(Exception):
    """Base class for all service errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"

    def __init__(self, detail: str, **context: Any):
        super().__init__(detail)
        self.detail = detail
        self.context = context


class DatabaseConnectionError(GraphRAGError):
    """ArcadeDB is unreachable, timed out, or returned a transport-level failure."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "database_unavailable"


class DatabaseQueryError(GraphRAGError):
    """ArcadeDB accepted the request but rejected the query."""

    status_code = status.HTTP_502_BAD_GATEWAY
    error_code = "database_query_failed"


class TenantNotFoundError(GraphRAGError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "tenant_not_found"

    def __init__(self, tenant_id: str):
        super().__init__(
            f"Tenant knowledge base for '{tenant_id}' does not exist or is not provisioned.",
            tenant_id=tenant_id,
        )


class TenantAccessDeniedError(GraphRAGError):
    """The credential is valid but not authorised for the requested tenant."""

    status_code = status.HTTP_403_FORBIDDEN
    error_code = "tenant_access_denied"


class AuthenticationError(GraphRAGError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "unauthenticated"


class RateLimitExceededError(GraphRAGError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error_code = "rate_limit_exceeded"

    def __init__(self, detail: str, retry_after_seconds: int = 60):
        super().__init__(detail, retry_after_seconds=retry_after_seconds)
        self.retry_after_seconds = retry_after_seconds


class SchemaValidationError(GraphRAGError):
    """A vertex label or edge type is not in the tenant's approved schema."""

    status_code = 422
    error_code = "schema_validation_failed"


class SecurityViolationError(GraphRAGError):
    """A potentially malicious payload was detected and rejected."""

    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "security_violation"


class EntityResolutionError(GraphRAGError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "entity_resolution_failed"


class ModelUnavailableError(GraphRAGError):
    """An embedding or reranking model could not be loaded."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_unavailable"


def _payload(exc: GraphRAGError, request: Request) -> Dict[str, Any]:
    return {
        "error": exc.error_code,
        "detail": exc.detail,
        "request_id": getattr(request.state, "request_id", None),
    }


async def graph_rag_exception_handler(request: Request, exc: GraphRAGError) -> JSONResponse:
    """Single handler for the whole taxonomy; logs server-side faults with context."""
    if exc.status_code >= 500:
        logger.error(
            "%s: %s | context=%s | request_id=%s",
            exc.error_code,
            exc.detail,
            exc.context,
            getattr(request.state, "request_id", None),
        )
    headers: Dict[str, str] = {}
    if isinstance(exc, RateLimitExceededError):
        headers["Retry-After"] = str(exc.retry_after_seconds)
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(exc, request),
        headers=headers or None,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internals or stack traces to the caller."""
    logger.exception(
        "Unhandled exception | request_id=%s", getattr(request.state, "request_id", None)
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "detail": "An unexpected internal error occurred.",
            "request_id": getattr(request.state, "request_id", None),
        },
    )
