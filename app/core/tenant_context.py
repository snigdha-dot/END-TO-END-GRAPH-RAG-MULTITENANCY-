"""Security Layer 2: async tenant context guard (plan section 2.1).

Every database call resolves its target database through this contextvar rather
than through a parameter passed down the call stack. A service that forgets to
thread `tenant_id` through cannot silently fall back to "no tenant" — it raises.

`contextvars` are per-task in asyncio, so concurrent requests for different
tenants cannot observe each other's context.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator, Optional

from app.core.exceptions import TenantAccessDeniedError
from app.core.tenant_schema import TenantGraphSchema, schema_registry


@dataclass(frozen=True)
class TenantContext:
    """The authenticated, verified identity of the caller for one request."""

    tenant_id: str
    api_key_id: str
    request_id: str
    user_id: Optional[str] = None
    auth_method: str = "api_key"
    scopes: tuple[str, ...] = ()

    @property
    def db_name(self) -> str:
        return f"tenant_{self.tenant_id.lower()}_kb"

    @property
    def schema(self) -> TenantGraphSchema:
        return schema_registry.get(self.tenant_id)

    def require_scope(self, scope: str) -> None:
        if self.scopes and scope not in self.scopes:
            raise TenantAccessDeniedError(
                f"Credential lacks required scope '{scope}'.",
                tenant_id=self.tenant_id,
                required_scope=scope,
            )


_tenant_context: ContextVar[Optional[TenantContext]] = ContextVar(
    "team_b_tenant_context", default=None
)


def set_tenant_context(ctx: TenantContext) -> Token:
    """Bind the verified tenant for the current async task."""
    return _tenant_context.set(ctx)


def reset_tenant_context(token: Token) -> None:
    _tenant_context.reset(token)


def get_tenant_context() -> TenantContext:
    """Return the bound tenant context, raising if none was established."""
    ctx = _tenant_context.get()
    if ctx is None:
        # Fail closed: an unscoped database call is a bug, not a broad query.
        raise TenantAccessDeniedError(
            "No tenant context is bound to this execution scope; refusing unscoped access."
        )
    return ctx


def current_tenant_id() -> str:
    return get_tenant_context().tenant_id


def try_get_tenant_context() -> Optional[TenantContext]:
    """Non-raising accessor, for logging paths that must not fail."""
    return _tenant_context.get()


@contextmanager
def tenant_scope(ctx: TenantContext) -> Iterator[TenantContext]:
    """Bind a tenant context for the duration of a block (tests, workers, scripts)."""
    token = set_tenant_context(ctx)
    try:
        yield ctx
    finally:
        reset_tenant_context(token)
