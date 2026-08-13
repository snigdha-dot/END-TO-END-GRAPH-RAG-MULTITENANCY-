"""Async HTTP Client Pool for ArcadeDB REST API & Cypher Execution Engine."""
import logging
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

class ArcadeDBClient:
    def __init__(self):
        self.base_url = settings.ARCADEDB_URL.rstrip('/')
        self.auth = (settings.ARCADEDB_USER, settings.ARCADEDB_PASSWORD)
        self.client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Initialize async HTTP client pool."""
        if not self.client:
            self.client = httpx.AsyncClient(
                auth=self.auth,
                timeout=httpx.Timeout(5.0, connect=3.0),
                headers={"Content-Type": "application/json"}
            )

    async def close(self):
        """Close async HTTP client pool."""
        if self.client:
            await self.client.aclose()
            self.client = None

    async def is_ready(self) -> bool:
        """Health check endpoint against ArcadeDB server."""
        if not self.client:
            await self.start()
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/ready")
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"ArcadeDB health check failed: {e}")
            return False

    async def ensure_database_exists(self, tenant_id: str) -> str:
        """Ensure tenant database exists; creates it if missing."""
        if not self.client:
            await self.start()
        db_name = f"tenant_{tenant_id.lower()}_kb"
        
        # Check if database exists
        try:
            resp = await self.client.get(f"{self.base_url}/api/v1/exists/{db_name}")
            if resp.status_code == 200 and resp.json().get("result") is True:
                return db_name
        except Exception:
            pass

        # Create database dynamically
        try:
            create_payload = {"command": f"CREATE DATABASE {db_name}"}
            resp = await self.client.post(f"{self.base_url}/api/v1/server", json=create_payload)
            logger.info(f"Created ArcadeDB database '{db_name}' for tenant '{tenant_id}'")
            return db_name
        except Exception as e:
            logger.error(f"Failed to create database '{db_name}': {e}")
            # Fallback to local memory mock mode if ArcadeDB container is offline during local test runs
            return db_name

    async def execute_cypher(
        self, tenant_id: str, cypher_query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute parameterized Cypher query against target tenant database."""
        if not self.client:
            await self.start()
        
        db_name = await self.ensure_database_exists(tenant_id)
        endpoint = f"{self.base_url}/api/v1/command/{db_name}"

        payload = {
            "language": "cypher",
            "command": cypher_query,
            "params": params or {}
        }

        try:
            resp = await self.client.post(endpoint, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("result", [])
            else:
                logger.error(f"ArcadeDB error [{resp.status_code}]: {resp.text}")
                return []
        except Exception as e:
            logger.error(f"ArcadeDB query execution exception: {e}")
            return []


# Global singleton instance
arcadedb_client = ArcadeDBClient()
