"""The service must never disguise a failure as an empty result.

This is the regression guard for the original P0 defect: the client caught every
exception and returned `[]`, and retrieval turned `[]` into a synthetic passage.
The result was HTTP 200 with plausible telemetry while the database was entirely
unreachable â€” indistinguishable, to a caller, from a working system.
"""
from __future__ import annotations

import httpx
import pytest

from app.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    TenantNotFoundError,
)
from app.services.arcadedb_client import ArcadeDBClient


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or str(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def client() -> ArcadeDBClient:
    c = ArcadeDBClient()
    c._known_databases.add("tenant_movies_bot_kb")
    return c


@pytest.mark.asyncio
async def test_connection_failure_raises_not_returns_empty(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(client, "_get_client", _stub_client(boom))

    with pytest.raises(DatabaseConnectionError):
        await client.execute_cypher("MATCH (n) RETURN n", tenant_id="movies_bot")


@pytest.mark.asyncio
async def test_timeout_raises_not_returns_empty(client, monkeypatch):
    async def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(client, "_get_client", _stub_client(timeout))

    with pytest.raises(DatabaseConnectionError):
        await client.execute_cypher("MATCH (n) RETURN n", tenant_id="movies_bot")


@pytest.mark.asyncio
async def test_query_rejection_raises_query_error(client, monkeypatch):
    async def rejected(*args, **kwargs):
        return _FakeResponse(400, text="syntax error near MATCH")

    monkeypatch.setattr(client, "_get_client", _stub_client(rejected))

    with pytest.raises(DatabaseQueryError):
        await client.execute_cypher("MATCH bad syntax", tenant_id="movies_bot")


@pytest.mark.asyncio
async def test_auth_failure_raises_connection_error(client, monkeypatch):
    async def unauthorized(*args, **kwargs):
        return _FakeResponse(401, text="unauthorized")

    monkeypatch.setattr(client, "_get_client", _stub_client(unauthorized))

    with pytest.raises(DatabaseConnectionError):
        await client.execute_cypher("MATCH (n) RETURN n", tenant_id="movies_bot")


@pytest.mark.asyncio
async def test_genuinely_empty_result_returns_empty_list(client, monkeypatch):
    """The one case that legitimately yields []: the query ran and matched nothing."""

    async def empty(*args, **kwargs):
        return _FakeResponse(200, {"result": []})

    monkeypatch.setattr(client, "_get_client", _stub_client(empty))

    result = await client.execute_cypher("MATCH (n) RETURN n", tenant_id="movies_bot")
    assert result == []


@pytest.mark.asyncio
async def test_successful_result_is_returned(client, monkeypatch):
    async def ok(*args, **kwargs):
        return _FakeResponse(200, {"result": [{"entity_id": "canon_inception"}]})

    monkeypatch.setattr(client, "_get_client", _stub_client(ok))

    result = await client.execute_cypher("MATCH (n) RETURN n", tenant_id="movies_bot")
    assert result == [{"entity_id": "canon_inception"}]


@pytest.mark.asyncio
async def test_unprovisioned_tenant_raises_not_found(monkeypatch):
    """A typo'd tenant must 404, not silently create an empty knowledge base."""
    c = ArcadeDBClient()

    async def not_exists(*args, **kwargs):
        return _FakeResponse(200, {"result": False})

    monkeypatch.setattr(c, "_get_client", _stub_client(not_exists))

    with pytest.raises(TenantNotFoundError):
        await c.execute_cypher("MATCH (n) RETURN n", tenant_id="typo_tenant")


@pytest.mark.asyncio
async def test_unscoped_query_is_refused():
    """No tenant context and no explicit tenant means no query."""
    c = ArcadeDBClient()
    with pytest.raises(DatabaseQueryError, match="unscoped"):
        await c.execute_cypher("MATCH (n) RETURN n")


def _stub_client(handler):
    """Return a factory producing a stand-in httpx.AsyncClient with scripted calls.

    Returns a fresh coroutine per call, since `_get_client` is awaited more than
    once per query (existence check, then the command itself).
    """

    class _Stub:
        async def request(self, *args, **kwargs):
            return await handler(*args, **kwargs)

        async def get(self, *args, **kwargs):
            return await handler(*args, **kwargs)

    async def _get():
        return _Stub()

    return _get

