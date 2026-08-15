"""Async HTTP client pool for ArcadeDB REST API & Cypher execution.

Core contract: **a failure is never an empty result.**

Transport failures raise `DatabaseConnectionError`; query rejections raise
`DatabaseQueryError`. Only a query that ArcadeDB executed successfully and that
matched nothing returns `[]`. Without this distinction the retrieval fallback
cannot tell "nothing indexed" from "database down", and the service returns
HTTP 200 with synthetic context while completely broken.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence

import httpx

from app.core.config import settings
from app.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    TenantNotFoundError,
)
from app.core.security import TenantIdValidator
from app.core.tenant_context import try_get_tenant_context

logger = logging.getLogger(__name__)

# Transient conditions worth one retry; anything else fails immediately.
_RETRYABLE_STATUS = {502, 503, 504}


class ArcadeDBClient:
    """Connection-pooled ArcadeDB REST client with strict tenant scoping."""

    def __init__(self) -> None:
        self.base_url = settings.ARCADEDB_URL.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        # Databases confirmed to exist, so we stop re-checking on every query.
        self._known_databases: set[str] = set()
        # None = not yet probed. Set on first batch attempt so an unsupported
        # endpoint costs one 404 rather than one per batch.
        self._batch_endpoint_available: Optional[bool] = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        """Initialize the pooled HTTP client."""
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    auth=(settings.ARCADEDB_USER, settings.ARCADEDB_PASSWORD),
                    timeout=httpx.Timeout(
                        settings.query_timeout_seconds,
                        connect=settings.connect_timeout_seconds,
                    ),
                    limits=httpx.Limits(
                        max_connections=settings.ARCADEDB_MAX_CONNECTIONS,
                        max_keepalive_connections=settings.ARCADEDB_MAX_KEEPALIVE,
                    ),
                    headers={"Content-Type": "application/json"},
                )
                logger.info("ArcadeDB client pool initialized for %s", self.base_url)

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
                self._known_databases.clear()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        assert self._client is not None
        return self._client

    # ------------------------------------------------------------------ health
    async def is_ready(self) -> bool:
        """Liveness probe. Returns a boolean by design — never raises."""
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/v1/ready")
            # ArcadeDB answers 204 No Content when healthy.
            return resp.status_code in (200, 204)
        except Exception as exc:  # noqa: BLE001 - probe must not propagate
            logger.warning("ArcadeDB readiness probe failed: %s", exc)
            return False

    # ------------------------------------------------------------------ requests
    async def _request(
        self, method: str, url: str, *, json_body: Optional[Dict[str, Any]] = None,
        retry: bool = True, timeout_ms: Optional[int] = None,
    ) -> httpx.Response:
        """Perform an HTTP call, translating transport faults into typed errors."""
        client = await self._get_client()
        attempts = 2 if retry else 1
        last_exc: Optional[Exception] = None
        effective_timeout = timeout_ms or settings.ARCADEDB_QUERY_TIMEOUT_MS
        request_timeout = httpx.Timeout(
            effective_timeout / 1000.0, connect=settings.connect_timeout_seconds
        )

        for attempt in range(attempts):
            try:
                resp = await client.request(method, url, json=json_body, timeout=request_timeout)
                if resp.status_code in _RETRYABLE_STATUS and attempt < attempts - 1:
                    await asyncio.sleep(0.15 * (attempt + 1))
                    continue
                return resp
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(0.15 * (attempt + 1))
                    continue
                raise DatabaseConnectionError(
                    f"ArcadeDB request timed out after {effective_timeout}ms.",
                    url=url,
                ) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(0.15 * (attempt + 1))
                    continue
                raise DatabaseConnectionError(
                    f"ArcadeDB is unreachable: {exc}", url=url
                ) from exc

        raise DatabaseConnectionError(
            f"ArcadeDB request failed: {last_exc}", url=url
        )

    # ------------------------------------------------------------------ databases
    async def database_exists(self, db_name: str) -> bool:
        resp = await self._request("GET", f"{self.base_url}/api/v1/exists/{db_name}")
        if resp.status_code == 200:
            try:
                return bool(resp.json().get("result", False))
            except Exception:  # noqa: BLE001 - malformed body means "cannot confirm"
                return False
        if resp.status_code == 404:
            return False
        raise DatabaseQueryError(
            f"Unexpected response checking database existence: HTTP {resp.status_code}",
            db_name=db_name,
            body=resp.text[:400],
        )

    async def create_database(self, tenant_id: str) -> str:
        """Explicitly provision a tenant database. Admin operation only."""
        tenant = TenantIdValidator.validate(tenant_id)
        db_name = f"tenant_{tenant}_kb"

        if await self.database_exists(db_name):
            self._known_databases.add(db_name)
            return db_name

        resp = await self._request(
            "POST",
            f"{self.base_url}/api/v1/server",
            json_body={"command": f"create database {db_name}"},
        )
        if resp.status_code not in (200, 201, 204):
            raise DatabaseQueryError(
                f"Failed to create database '{db_name}': HTTP {resp.status_code}",
                db_name=db_name,
                body=resp.text[:400],
            )
        self._known_databases.add(db_name)
        logger.info("Provisioned ArcadeDB database '%s' for tenant '%s'", db_name, tenant)
        return db_name

    async def drop_database(self, tenant_id: str) -> None:
        """Delete a tenant database. Used by integration tests for teardown."""
        tenant = TenantIdValidator.validate(tenant_id)
        db_name = f"tenant_{tenant}_kb"
        resp = await self._request(
            "POST",
            f"{self.base_url}/api/v1/server",
            json_body={"command": f"drop database {db_name}"},
        )
        if resp.status_code not in (200, 204, 404):
            raise DatabaseQueryError(
                f"Failed to drop database '{db_name}': HTTP {resp.status_code}",
                db_name=db_name,
            )
        self._known_databases.discard(db_name)

    async def resolve_database(self, tenant_id: str) -> str:
        """Map a tenant to its database, verifying it exists.

        Never creates as a side effect of a read: unbounded auto-provisioning turns
        a loop of random tenant ids into a disk-exhaustion vector, and it masks
        typos as empty knowledge bases.
        """
        tenant = TenantIdValidator.validate(tenant_id)
        db_name = f"tenant_{tenant}_kb"

        if db_name in self._known_databases:
            return db_name

        if await self.database_exists(db_name):
            self._known_databases.add(db_name)
            return db_name

        if settings.ALLOW_TENANT_AUTOPROVISION:
            logger.warning(
                "Auto-provisioning database for tenant '%s' (development mode only)", tenant
            )
            return await self.create_database(tenant)

        raise TenantNotFoundError(tenant)

    # ------------------------------------------------------------------ queries
    def _extract_result(self, resp: httpx.Response, *, db_name: str, query: str) -> List[Dict[str, Any]]:
        """Translate an ArcadeDB response into rows, or raise a typed error."""
        if resp.status_code == 200:
            try:
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise DatabaseQueryError(
                    "ArcadeDB returned a non-JSON success response.", db_name=db_name
                ) from exc
            result = payload.get("result", [])
            return result if isinstance(result, list) else [result]

        if resp.status_code == 404:
            raise TenantNotFoundError(db_name.removeprefix("tenant_").removesuffix("_kb"))

        if resp.status_code in (401, 403):
            raise DatabaseConnectionError(
                "ArcadeDB rejected the service credentials.", db_name=db_name
            )

        if resp.status_code in _RETRYABLE_STATUS:
            raise DatabaseConnectionError(
                f"ArcadeDB is unavailable: HTTP {resp.status_code}", db_name=db_name
            )

        # 400/500: the server ran and refused the query. Log the query, not the params.
        logger.error(
            "ArcadeDB rejected query on '%s' [HTTP %s]: %s | query=%s",
            db_name, resp.status_code, resp.text[:500], query[:300],
        )
        raise DatabaseQueryError(
            f"ArcadeDB rejected the query: HTTP {resp.status_code}",
            db_name=db_name,
            body=resp.text[:400],
        )

    async def execute_cypher(
        self,
        cypher_query: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        tenant_id: Optional[str] = None,
        language: str = "cypher",
        timeout_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a parameterized query against the bound tenant's database.

        `tenant_id` defaults to the contextvar bound by the auth layer, so a caller
        cannot accidentally issue an unscoped query.
        """
        if tenant_id is None:
            ctx = try_get_tenant_context()
            if ctx is None:
                raise DatabaseQueryError(
                    "No tenant context bound and no tenant_id supplied; refusing unscoped query."
                )
            tenant_id = ctx.tenant_id

        db_name = await self.resolve_database(tenant_id)
        resp = await self._request(
            "POST",
            f"{self.base_url}/api/v1/command/{db_name}",
            json_body={
                "language": language,
                "command": cypher_query,
                "params": params or {},
                "limit": settings.MAX_TRAVERSAL_NODES,
            },
            timeout_ms=timeout_ms,
        )
        return self._extract_result(resp, db_name=db_name, query=cypher_query)

    async def execute_sql(
        self, sql: str, params: Optional[Dict[str, Any]] = None, *,
        tenant_id: Optional[str] = None, timeout_ms: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute ArcadeDB SQL — needed for schema/index DDL, which Cypher lacks."""
        return await self.execute_cypher(
            sql, params, tenant_id=tenant_id, language="sql", timeout_ms=timeout_ms
        )

    async def execute_batch(
        self,
        statements: Sequence[Dict[str, Any]],
        *,
        tenant_id: Optional[str] = None,
        language: str = "cypher",
    ) -> int:
        """Execute many write statements, batching where the server supports it.

        One HTTP round-trip per vertex made ingestion O(n) network calls. Where a
        server-side batch mechanism exists it is used; otherwise statements are
        executed sequentially so ingestion still completes correctly.

        Note on ArcadeDB 24.x: `/api/v1/batch/{db}` is not present in all builds,
        and the `sqlscript` language does not accept bound parameters. Since
        parameter binding is a security requirement here, not a convenience, the
        sequential path is used rather than interpolating values into a script.
        """
        if not statements:
            return 0

        if tenant_id is None:
            ctx = try_get_tenant_context()
            if ctx is None:
                raise DatabaseQueryError("No tenant context bound; refusing unscoped write.")
            tenant_id = ctx.tenant_id

        db_name = await self.resolve_database(tenant_id)
        executed = 0

        if self._batch_endpoint_available is not False:
            batch_size = max(1, settings.ARCADEDB_WRITE_BATCH_SIZE)
            for start in range(0, len(statements), batch_size):
                chunk = statements[start : start + batch_size]
                commands = [
                    {
                        "language": language,
                        "command": stmt["command"],
                        "params": stmt.get("params", {}),
                    }
                    for stmt in chunk
                ]
                resp = await self._request(
                    "POST",
                    f"{self.base_url}/api/v1/batch/{db_name}",
                    json_body={"operations": commands},
                    retry=False,
                    timeout_ms=settings.ARCADEDB_WRITE_TIMEOUT_MS,
                )
                if resp.status_code in (200, 201, 204):
                    self._batch_endpoint_available = True
                    executed += len(chunk)
                    continue

                # Probe once, then remember, so we do not pay a 404 per batch.
                logger.info(
                    "Batch endpoint unavailable (HTTP %s); using sequential writes.",
                    resp.status_code,
                )
                self._batch_endpoint_available = False
                break

        if executed == 0:
            executed = await self._execute_concurrent(statements, tenant_id, language)

        return executed

    async def _execute_concurrent(
        self, statements: Sequence[Dict[str, Any]], tenant_id: str, language: str
    ) -> int:
        """Run write statements concurrently against a bounded connection pool.

        With no server-side batch endpoint, writes go one statement per request.
        Issued serially that is ~120ms of round-trip latency each and dominated
        ingestion: 2,108 statements for two documents took 13 minutes, almost all
        of it waiting on the network rather than on the database.

        The work is I/O-bound, so a bounded semaphore overlaps the waiting. The
        bound matters: unbounded concurrency would exhaust the pool and starve
        concurrent retrieval traffic.
        """
        semaphore = asyncio.Semaphore(settings.ARCADEDB_WRITE_CONCURRENCY)
        failures: List[Exception] = []

        async def run_one(stmt: Dict[str, Any]) -> bool:
            async with semaphore:
                try:
                    await self.execute_cypher(
                        stmt["command"],
                        stmt.get("params", {}),
                        tenant_id=tenant_id,
                        language=language,
                        timeout_ms=settings.ARCADEDB_WRITE_TIMEOUT_MS,
                    )
                    return True
                except DatabaseConnectionError as exc:
                    # A single-node server under write pressure returns 503. That is
                    # back-pressure, not a dead database, so back off and retry once
                    # before treating it as fatal.
                    await asyncio.sleep(0.5)
                    try:
                        await self.execute_cypher(
                            stmt["command"],
                            stmt.get("params", {}),
                            tenant_id=tenant_id,
                            language=language,
                            timeout_ms=settings.ARCADEDB_WRITE_TIMEOUT_MS,
                        )
                        return True
                    except (DatabaseQueryError, DatabaseConnectionError) as retry_exc:
                        failures.append(retry_exc)
                        return False
                except DatabaseQueryError as exc:
                    failures.append(exc)
                    return False

        # Chunked so a huge document does not create one task per statement at once.
        executed = 0
        window = max(1, settings.ARCADEDB_WRITE_CONCURRENCY * 20)
        for start in range(0, len(statements), window):
            batch = statements[start : start + window]
            results = await asyncio.gather(*(run_one(s) for s in batch))
            executed += sum(1 for ok in results if ok)

            # Sustained connection failures (already retried once each) mean the
            # database is genuinely gone; continuing would produce a silently
            # partial graph, which is worse than failing loudly.
            connection_failures = [
                f for f in failures if isinstance(f, DatabaseConnectionError)
            ]
            if len(connection_failures) > max(5, len(statements) // 20):
                raise connection_failures[0]

        if failures:
            logger.warning(
                "%d of %d write statements failed for tenant '%s'; first error: %s",
                len(failures), len(statements), tenant_id, failures[0].detail,
            )

        return executed


# Global singleton; lifecycle is managed by the FastAPI lifespan handler.
arcadedb_client = ArcadeDBClient()
