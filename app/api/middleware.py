"""Security, rate limiting, and audit middleware (plan section 6: middleware.py).

Ordering matters. Starlette runs middleware in reverse registration order, so these
are added outermost-first in `main.py`:

    RequestContext  -> assigns request_id, starts the clock
    SecurityHeaders -> response hardening
    BodySizeLimit   -> reject oversized payloads before parsing
    RateLimit       -> shed load before touching the database
    AuditLog        -> record the outcome
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Awaitable, Callable, Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.core.security import api_key_fingerprint
from app.core.tenant_context import try_get_tenant_context

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("team_b.audit")

_Next = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id and elapsed-time budget to every request."""

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        incoming = request.headers.get(settings.REQUEST_ID_HEADER)
        # Only trust an upstream id if it looks like one; otherwise it is log-injection.
        request_id = (
            incoming
            if incoming and len(incoming) <= 64 and incoming.replace("-", "").isalnum()
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        response = await call_next(request)
        response.headers[settings.REQUEST_ID_HEADER] = request_id
        elapsed_ms = (time.perf_counter() - request.state.start_time) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Apply standard hardening headers to every response."""

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault(
            "Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'"
        )
        if settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies using Content-Length, before JSON parsing."""

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "payload_too_large",
                            "detail": (
                                f"Request body exceeds the "
                                f"{settings.MAX_REQUEST_BODY_BYTES} byte limit."
                            ),
                            "request_id": getattr(request.state, "request_id", None),
                        },
                    )
            except ValueError:
                pass
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window per-credential rate limiter.

    In-process only: with multiple workers each holds its own window, so the
    effective limit is per-worker. That is an intentional simple default — a
    multi-instance deployment should move this to Redis. It still contains the
    failure mode it exists for: one client looping and exhausting the DB pool.
    """

    def __init__(self, app):
        super().__init__(app)
        self._windows: Dict[str, Deque[float]] = defaultdict(deque)
        self._window_seconds = 60.0

    def _client_id(self, request: Request) -> str:
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if api_key:
            return f"key:{api_key_fingerprint(api_key)}"
        client = request.client
        return f"ip:{client.host if client else 'unknown'}"

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        if not settings.RATE_LIMIT_ENABLED or request.url.path in ("/health", "/ready", "/live"):
            return await call_next(request)

        client_id = self._client_id(request)
        now = time.monotonic()
        window = self._windows[client_id]

        while window and now - window[0] > self._window_seconds:
            window.popleft()

        limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE + settings.RATE_LIMIT_BURST
        if len(window) >= limit:
            retry_after = max(1, int(self._window_seconds - (now - window[0])))
            audit_logger.warning(
                "rate_limit_exceeded client=%s path=%s request_id=%s",
                client_id, request.url.path, getattr(request.state, "request_id", None),
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": (
                        f"Rate limit of {settings.RATE_LIMIT_REQUESTS_PER_MINUTE} "
                        "requests/minute exceeded."
                    ),
                    "request_id": getattr(request.state, "request_id", None),
                },
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)

        # Bound memory: drop windows that have fully aged out.
        if len(self._windows) > 10_000:
            stale = [k for k, w in self._windows.items() if not w or now - w[-1] > self._window_seconds]
            for k in stale:
                del self._windows[k]

        response = await call_next(request)
        remaining = max(0, limit - len(window))
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Structured audit trail: who queried which tenant, when, and with what outcome.

    Deliberately records the API key *fingerprint* and never the key, the query text,
    or retrieved content — an audit log should not become a second copy of tenant data.
    """

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        start = time.perf_counter()
        request_id = getattr(request.state, "request_id", None)

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            audit_logger.exception(
                "request_failed method=%s path=%s duration_ms=%.2f request_id=%s",
                request.method, request.url.path, duration_ms, request_id,
            )
            raise

        duration_ms = (time.perf_counter() - start) * 1000
        ctx = getattr(request.state, "tenant_context", None) or try_get_tenant_context()

        if request.url.path not in ("/health", "/ready", "/live", "/metrics"):
            audit_logger.info(
                "method=%s path=%s status=%s tenant=%s key_id=%s user=%s auth=%s "
                "duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                response.status_code,
                ctx.tenant_id if ctx else "-",
                ctx.api_key_id if ctx else "-",
                (ctx.user_id if ctx and ctx.user_id else "-"),
                ctx.auth_method if ctx else "-",
                duration_ms,
                request_id,
            )
        return response
