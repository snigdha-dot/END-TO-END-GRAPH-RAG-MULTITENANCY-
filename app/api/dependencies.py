"""Security Layer 1: authentication and tenant authorization (plan section 2.1).

The controlling principle: **the credential decides the tenant.** A caller does not
get to declare which knowledge base it reads. An API key maps to exactly one tenant;
a JWT carries a signed `tenant_id` claim. If a request also sends `X-Tenant-ID`, it
is treated as an assertion to be *checked* against the credential, and a mismatch is
a 403 — not a silent override.

The previous implementation validated only that a key existed in a set, which meant
any valid key could read any tenant's data by changing one header.
"""
from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Depends, Header, Request

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationError,
    TenantAccessDeniedError,
    TenantNotFoundError,
)
from app.core.security import JWTVerifier, TenantIdValidator, api_key_fingerprint
from app.core.tenant_context import TenantContext, set_tenant_context
from app.core.tenant_schema import schema_registry

logger = logging.getLogger(__name__)


def _resolve_api_key_tenant(api_key: str) -> str:
    """Constant-time lookup of the tenant an API key is bound to."""
    # Compare against every configured key so a wrong key costs the same time as a
    # right one, denying a timing oracle over the key set.
    matched: Optional[str] = None
    for configured_key, tenant in settings.API_KEY_TENANT_MAP.items():
        if hmac.compare_digest(api_key, configured_key):
            matched = tenant
    if matched is None:
        raise AuthenticationError("Invalid or unknown API key.")
    return matched


async def authenticate_request(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias=settings.API_KEY_HEADER),
    x_tenant_id: Optional[str] = Header(default=None, alias=settings.TENANT_HEADER),
    authorization: Optional[str] = Header(default=None),
) -> TenantContext:
    """Authenticate the caller and bind a verified tenant context.

    Resolution order:
      1. API key -> its single authorized tenant (always required).
      2. If a JWT is present (or required), verify it and cross-check its
         `tenant_id` claim against the API key's tenant.
      3. If `X-Tenant-ID` is present, verify it agrees with the above.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not x_api_key:
        raise AuthenticationError(f"Missing required {settings.API_KEY_HEADER} header.")

    tenant_from_key = _resolve_api_key_tenant(x_api_key)
    key_id = api_key_fingerprint(x_api_key)

    user_id: Optional[str] = None
    scopes: tuple[str, ...] = ()
    auth_method = "api_key"

    bearer: Optional[str] = None
    if authorization and authorization.lower().startswith("bearer "):
        bearer = authorization[7:].strip()

    if settings.JWT_REQUIRED and not bearer:
        raise AuthenticationError("A Bearer JWT is required but was not supplied.")

    if bearer and settings.JWT_ENABLED:
        claims = JWTVerifier.verify(bearer)
        tenant_from_jwt = TenantIdValidator.validate(str(claims["tenant_id"]))

        # The signed claim and the API key must agree. Disagreement means a token was
        # replayed against the wrong chatbot's credential.
        if tenant_from_jwt != tenant_from_key:
            logger.warning(
                "Tenant mismatch between JWT and API key | key_id=%s jwt_tenant=%s "
                "key_tenant=%s request_id=%s",
                key_id, tenant_from_jwt, tenant_from_key, request_id,
            )
            raise TenantAccessDeniedError(
                "JWT tenant claim does not match the tenant bound to this API key.",
                tenant_id=tenant_from_key,
            )

        user_id = str(claims["user_id"]) if claims.get("user_id") else None
        raw_scope = claims.get("scope") or claims.get("scopes") or []
        if isinstance(raw_scope, str):
            scopes = tuple(raw_scope.split())
        else:
            scopes = tuple(str(s) for s in raw_scope)
        auth_method = "api_key+jwt"

    tenant_id = tenant_from_key

    # A supplied tenant header is an assertion we verify, never an override.
    if x_tenant_id:
        asserted = TenantIdValidator.validate(x_tenant_id)
        if asserted != tenant_id:
            logger.warning(
                "Cross-tenant access attempt | key_id=%s asserted=%s authorized=%s request_id=%s",
                key_id, asserted, tenant_id, request_id,
            )
            raise TenantAccessDeniedError(
                f"This credential is not authorized for tenant '{asserted}'.",
                tenant_id=asserted,
                authorized_tenant=tenant_id,
            )

    ctx = TenantContext(
        tenant_id=tenant_id,
        api_key_id=key_id,
        request_id=request_id,
        user_id=user_id,
        auth_method=auth_method,
        scopes=scopes,
    )

    # Bind for the lifetime of this request so every downstream DB call is scoped.
    set_tenant_context(ctx)
    request.state.tenant_context = ctx
    return ctx


async def require_retrieval_scope(
    ctx: TenantContext = Depends(authenticate_request),
) -> TenantContext:
    ctx.require_scope("retrieval:read")
    return ctx


async def require_ingestion_scope(
    ctx: TenantContext = Depends(authenticate_request),
) -> TenantContext:
    ctx.require_scope("ingestion:write")
    return ctx


async def require_provisioned_tenant(
    ctx: TenantContext = Depends(authenticate_request),
) -> TenantContext:
    """Reject early when the tenant has a credential but no curated schema."""
    if not schema_registry.has(ctx.tenant_id) and not settings.ALLOW_TENANT_AUTOPROVISION:
        raise TenantNotFoundError(ctx.tenant_id)
    return ctx


def verify_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> str:
    """Guard for provisioning endpoints.

    Admin actions are gated on a dedicated key that is never issued to Team A.
    """
    if not settings.ADMIN_API_KEY:
        raise TenantAccessDeniedError("Administrative endpoints are disabled.")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.ADMIN_API_KEY):
        raise AuthenticationError("Invalid or missing administrative key.")
    return "admin"
