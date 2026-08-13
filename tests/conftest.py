"""Shared pytest fixtures and test configuration."""
from __future__ import annotations

import os
from typing import Dict, Iterator

import pytest

# Configure the test environment before app modules import settings.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("ALLOW_TENANT_AUTOPROVISION", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")

from app.core.config import settings  # noqa: E402
from app.core.tenant_context import TenantContext, tenant_scope  # noqa: E402

MOVIES_KEY = "test_movies_key_0123456789abcdef"
AI_TRENDS_KEY = "test_ai_trends_key_0123456789abc"
ADMIN_KEY = "test_admin_key_0123456789abcdefgh"


@pytest.fixture(autouse=True)
def _test_credentials(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Bind deterministic per-tenant API keys for every test."""
    monkeypatch.setattr(
        settings,
        "API_KEY_TENANT_MAP",
        {MOVIES_KEY: "movies_bot", AI_TRENDS_KEY: "ai_trends_bot"},
        raising=False,
    )
    monkeypatch.setattr(settings, "ADMIN_API_KEY", ADMIN_KEY, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False, raising=False)
    yield


@pytest.fixture
def movies_headers() -> Dict[str, str]:
    return {"X-API-Key": MOVIES_KEY, "Content-Type": "application/json"}


@pytest.fixture
def ai_trends_headers() -> Dict[str, str]:
    return {"X-API-Key": AI_TRENDS_KEY, "Content-Type": "application/json"}


@pytest.fixture
def admin_headers() -> Dict[str, str]:
    return {"X-Admin-Key": ADMIN_KEY, "Content-Type": "application/json"}


@pytest.fixture
def movies_context() -> TenantContext:
    return TenantContext(
        tenant_id="movies_bot", api_key_id="test", request_id="test-request"
    )


@pytest.fixture
def ai_trends_context() -> TenantContext:
    return TenantContext(
        tenant_id="ai_trends_bot", api_key_id="test", request_id="test-request"
    )


@pytest.fixture
def client():
    """TestClient with exceptions surfaced as responses, matching production."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def bound_movies_tenant(movies_context: TenantContext) -> Iterator[TenantContext]:
    with tenant_scope(movies_context) as ctx:
        yield ctx
