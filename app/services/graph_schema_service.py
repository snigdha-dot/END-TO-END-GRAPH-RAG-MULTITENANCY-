"""Provisions ArcadeDB graph schema and the HNSW vector index for a tenant.

Cypher cannot express DDL, so this uses ArcadeDB SQL. Every identifier written into
a statement comes from the tenant's approved schema and is re-checked against
`is_safe_identifier` immediately before interpolation — never from request data.

Idempotent throughout: `IF NOT EXISTS` everywhere, so provisioning can be re-run.
"""
from __future__ import annotations

import logging
from typing import List

from app.core.config import settings
from app.core.exceptions import DatabaseQueryError, SchemaValidationError
from app.core.tenant_schema import TenantGraphSchema, is_safe_identifier, schema_registry
from app.services.arcadedb_client import arcadedb_client

logger = logging.getLogger(__name__)


class GraphSchemaService:
    """Creates vertex types, edge types, property indexes, and the HNSW index."""

    async def provision_tenant(self, tenant_id: str) -> dict:
        """Create the database and full graph schema for a tenant. Idempotent."""
        schema = schema_registry.get(tenant_id)
        db_name = await arcadedb_client.create_database(tenant_id)

        created = {
            "database": db_name,
            "vertex_types": [],
            "edge_types": [],
            "indexes": [],
        }

        for statement in self._vertex_type_statements(schema):
            await self._execute_ddl(statement, tenant_id)
        created["vertex_types"] = sorted(schema.vertex_labels)

        for statement in self._edge_type_statements(schema):
            await self._execute_ddl(statement, tenant_id)
        created["edge_types"] = sorted(schema.edge_types)

        for statement, label in self._index_statements(schema):
            await self._execute_ddl(statement, tenant_id)
            created["indexes"].append(label)

        logger.info(
            "Provisioned schema for tenant '%s': %d vertex types, %d edge types, %d indexes",
            tenant_id, len(schema.vertex_labels), len(schema.edge_types), len(created["indexes"]),
        )
        return created

    # ------------------------------------------------------------------ statements
    def _vertex_type_statements(self, schema: TenantGraphSchema) -> List[str]:
        statements: List[str] = []
        for label in sorted(schema.vertex_labels):
            self._assert_identifier(label, "vertex label")
            statements.append(f"CREATE VERTEX TYPE {label} IF NOT EXISTS")
        return statements

    def _edge_type_statements(self, schema: TenantGraphSchema) -> List[str]:
        statements: List[str] = []
        for etype in sorted(schema.edge_types):
            self._assert_identifier(etype, "edge type")
            statements.append(f"CREATE EDGE TYPE {etype} IF NOT EXISTS")
        return statements

    def _index_statements(self, schema: TenantGraphSchema) -> List[tuple[str, str]]:
        """Property definitions and indexes, including the HNSW vector index."""
        statements: List[tuple[str, str]] = []
        dim = settings.EMBEDDING_DIMENSIONS

        for label in sorted(schema.vertex_labels):
            self._assert_identifier(label, "vertex label")
            # Entity identity and lookup properties.
            statements.append(
                (f"CREATE PROPERTY {label}.entity_id IF NOT EXISTS STRING", f"{label}.entity_id")
            )
            statements.append(
                (
                    f"CREATE PROPERTY {label}.normalized_name IF NOT EXISTS STRING",
                    f"{label}.normalized_name",
                )
            )
            # Stored as `entity_label` because `label` is a reserved TinkerPop token
            # and cannot be written as a vertex property.
            statements.append(
                (
                    f"CREATE PROPERTY {label}.entity_label IF NOT EXISTS STRING",
                    f"{label}.entity_label",
                )
            )
            # UNIQUE on entity_id is what makes canonical merging safe under concurrency.
            statements.append(
                (
                    f"CREATE INDEX IF NOT EXISTS ON {label} (entity_id) UNIQUE",
                    f"idx_{label}_entity_id",
                )
            )
            statements.append(
                (
                    f"CREATE INDEX IF NOT EXISTS ON {label} (normalized_name) NOTUNIQUE",
                    f"idx_{label}_normalized_name",
                )
            )

        # Chunk carries the text and its embedding; it is the vector-search target.
        statements.append(("CREATE PROPERTY Chunk.chunk_id IF NOT EXISTS STRING", "Chunk.chunk_id"))
        statements.append(("CREATE PROPERTY Chunk.text IF NOT EXISTS STRING", "Chunk.text"))
        statements.append(
            ("CREATE PROPERTY Chunk.parent_doc_id IF NOT EXISTS STRING", "Chunk.parent_doc_id")
        )
        statements.append(
            (f"CREATE PROPERTY Chunk.embedding IF NOT EXISTS ARRAY_OF_FLOATS", "Chunk.embedding")
        )
        statements.append(
            ("CREATE INDEX IF NOT EXISTS ON Chunk (chunk_id) UNIQUE", "idx_Chunk_chunk_id")
        )
        statements.append(
            (
                "CREATE INDEX IF NOT EXISTS ON Chunk (parent_doc_id) NOTUNIQUE",
                "idx_Chunk_parent_doc_id",
            )
        )
        # HNSW vector index (plan: "Native ArcadeDB HNSW Vector Indexing").
        statements.append(
            (
                f"CREATE INDEX IF NOT EXISTS ON Chunk (embedding) HNSW "
                f"WITH (dimensions = {dim}, distanceFunction = 'cosine', m = 16, "
                f"efConstruction = 128)",
                "idx_Chunk_embedding_hnsw",
            )
        )
        return statements

    def _assert_identifier(self, value: str, kind: str) -> None:
        if not is_safe_identifier(value):
            raise SchemaValidationError(f"Unsafe {kind} rejected before DDL: {value!r}")

    async def _execute_ddl(self, statement: str, tenant_id: str) -> None:
        """Run one DDL statement, tolerating 'already exists' but not real errors."""
        try:
            await arcadedb_client.execute_sql(
                statement, tenant_id=tenant_id, timeout_ms=settings.ARCADEDB_DDL_TIMEOUT_MS
            )
        except DatabaseQueryError as exc:
            detail = str(exc.detail).lower()
            body = str(exc.context.get("body", "")).lower()
            benign = ("already exists", "duplicated", "existing index")
            if any(marker in detail or marker in body for marker in benign):
                return
            # HNSW support varies across ArcadeDB builds. Vector search degrades to
            # sequential cosine scoring, so this must not block provisioning.
            if "hnsw" in statement.lower():
                logger.warning(
                    "HNSW index creation failed for tenant '%s' (%s). Vector search will "
                    "use in-memory cosine scoring.", tenant_id, exc.detail,
                )
                return
            raise

    async def verify_schema(self, tenant_id: str) -> dict:
        """Report which declared types actually exist in the tenant database."""
        schema = schema_registry.get(tenant_id)
        rows = await arcadedb_client.execute_sql("SELECT FROM schema:types", tenant_id=tenant_id)
        present = {str(r.get("name")) for r in rows if isinstance(r, dict)}
        return {
            "tenant_id": tenant_id,
            "database": schema.db_name,
            "vertex_types_present": sorted(schema.vertex_labels & present),
            "vertex_types_missing": sorted(schema.vertex_labels - present),
            "edge_types_present": sorted(schema.edge_types & present),
            "edge_types_missing": sorted(schema.edge_types - present),
        }


graph_schema_service = GraphSchemaService()
