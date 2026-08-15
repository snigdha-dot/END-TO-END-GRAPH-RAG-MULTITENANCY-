"""Document ingestion orchestration (plan section 3).

    chunk -> embed -> extract -> resolve -> schema-validate -> batch write

Every graph write is parameterized and every label/edge identifier is validated
against the tenant's approved schema before it reaches a statement. Writes are
batched: the previous implementation issued one HTTP round-trip per vertex and
per edge, which made ingesting a real document take minutes.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.exceptions import SchemaValidationError
from app.core.security import CypherParameterizer
from app.core.telemetry import TelemetryTracker
from app.core.tenant_context import TenantContext
from app.core.tenant_schema import is_safe_identifier
from app.models.graph import Edge, Vertex
from app.services.arcadedb_client import arcadedb_client
from app.services.chunking_service import DocumentChunk, chunking_service
from app.services.embedding_service import embedding_service
from app.services.extraction_service import extraction_service
from app.services.resolution_service import resolution_service

logger = logging.getLogger(__name__)


class IngestionService:
    """Turns raw documents into an embedded, connected tenant knowledge graph."""

    async def ingest_document(
        self,
        ctx: TenantContext,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        telemetry = TelemetryTracker()
        schema = ctx.schema
        tenant_id = ctx.tenant_id

        safe_doc_id = CypherParameterizer.guard_user_text(doc_id, "doc_id")
        CypherParameterizer.guard_user_text(content[:2000], "content")

        # ---------------------------------------------------------- chunk
        t0 = time.perf_counter()
        chunks = chunking_service.chunk_document(safe_doc_id, content, metadata)
        telemetry.record_step_latency("chunking", (time.perf_counter() - t0) * 1000)

        if not chunks:
            return self._empty_result(safe_doc_id, tenant_id, telemetry)

        # ---------------------------------------------------------- embed
        t1 = time.perf_counter()
        texts = [c.contextual_text() for c in chunks]
        vectors = embedding_service.encode_batch(texts)
        embed_ms = (time.perf_counter() - t1) * 1000
        telemetry.record_step_latency("chunk_embedding", embed_ms)
        telemetry.record_model_call(
            step_name="chunk_embedding",
            model_name=embedding_service.model_label,
            prompt_tokens=sum(c.token_count for c in chunks),
            completion_tokens=0,
            duration_ms=embed_ms,
        )

        # ---------------------------------------------------------- extract
        t2 = time.perf_counter()
        all_vertices: List[Vertex] = []
        all_edges: List[Edge] = []
        mentions: List[Tuple[str, str]] = []  # (entity_id, chunk_id)

        for chunk in chunks:
            vertices, edges = extraction_service.extract_from_chunk(
                chunk.text, chunk.chunk_id, schema
            )
            all_vertices.extend(vertices)
            all_edges.extend(edges)
            for vertex in vertices:
                mentions.append((vertex.id, chunk.chunk_id))

        extract_ms = (time.perf_counter() - t2) * 1000
        telemetry.record_step_latency("entity_extraction", extract_ms)
        telemetry.record_model_call(
            step_name="entity_extraction",
            model_name=extraction_service.model_label,
            prompt_tokens=sum(c.token_count for c in chunks),
            completion_tokens=0,
            duration_ms=extract_ms,
        )

        # ---------------------------------------------------------- resolve
        t3 = time.perf_counter()
        resolved_vertices, resolved_edges = resolution_service.resolve_and_merge(
            all_vertices, all_edges, schema=schema
        )
        telemetry.record_step_latency("entity_resolution", (time.perf_counter() - t3) * 1000)

        # Remap mentions onto canonical ids and drop any that resolution discarded.
        canonical_ids = {v.id for v in resolved_vertices}
        alias_to_canonical: Dict[str, str] = {}
        for vertex in resolved_vertices:
            alias_to_canonical[vertex.id] = vertex.id
            for alias_norm in vertex.properties.get("normalized_aliases", []) or []:
                alias_to_canonical[
                    resolution_service.canonical_id_for(alias_norm, vertex.label)
                ] = vertex.id

        resolved_mentions = {
            (alias_to_canonical.get(eid, eid), cid)
            for eid, cid in mentions
            if alias_to_canonical.get(eid, eid) in canonical_ids
        }

        # ---------------------------------------------------------- validate
        validated_vertices = [
            v for v in resolved_vertices if self._validate_vertex(v, schema)
        ]
        validated_edges = [
            e for e in resolved_edges if self._validate_edge(e, schema, canonical_ids)
        ]
        rejected = (len(resolved_vertices) - len(validated_vertices)) + (
            len(resolved_edges) - len(validated_edges)
        )

        # ---------------------------------------------------------- write
        t4 = time.perf_counter()
        written = await self._write_graph(
            tenant_id, chunks, vectors, validated_vertices, validated_edges, resolved_mentions
        )
        telemetry.record_step_latency("graph_write", (time.perf_counter() - t4) * 1000)

        result = telemetry.finalize()
        return {
            "tenant_id": tenant_id,
            "doc_id": safe_doc_id,
            "chunks_created": len(chunks),
            "entities_extracted": len(validated_vertices),
            "relationships_created": len(validated_edges),
            "mentions_linked": len(resolved_mentions),
            "schema_rejections": rejected,
            "statements_executed": written,
            "embedding_model": embedding_service.model_label,
            "extraction_backend": extraction_service.active_backend,
            "telemetry": result,
            "status": "success",
            "execution_time_ms": result["latency_breakdown_ms"]["total_retrieval_latency"],
        }

    # ------------------------------------------------------------------ validation
    @staticmethod
    def _validate_vertex(vertex: Vertex, schema) -> bool:
        if not schema.validate_vertex_label(vertex.label) or not is_safe_identifier(vertex.label):
            logger.debug("Rejected vertex with label %r (not in tenant schema)", vertex.label)
            return False
        return True

    @staticmethod
    def _validate_edge(edge: Edge, schema, known_ids: set) -> bool:
        if not schema.validate_edge_type(edge.type) or not is_safe_identifier(edge.type):
            logger.debug("Rejected edge of type %r (not in tenant schema)", edge.type)
            return False
        if edge.confidence < settings.EDGE_CONFIDENCE_THRESHOLD:
            return False
        # An edge to an entity that resolution dropped would dangle.
        if edge.source not in known_ids or edge.target not in known_ids:
            return False
        return True

    # ------------------------------------------------------------------ writing
    async def _write_graph(
        self,
        tenant_id: str,
        chunks: List[DocumentChunk],
        vectors: List[List[float]],
        vertices: List[Vertex],
        edges: List[Edge],
        mentions: set,
    ) -> int:
        """Write chunks, entities, relationships, and mention links in batches."""
        statements: List[Dict[str, Any]] = []

        # Chunks carry the text and its embedding; upsert so re-ingest is idempotent.
        for chunk, vector in zip(chunks, vectors):
            statements.append(
                {
                    "command": (
                        "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                        "SET c.text = $text, c.parent_doc_id = $parent_doc_id, "
                        "c.chunk_index = $chunk_index, c.token_count = $token_count, "
                        "c.section_path = $section_path, c.embedding = $embedding, "
                        "c.prev_chunk_id = $prev_chunk_id, c.next_chunk_id = $next_chunk_id"
                    ),
                    "params": {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "parent_doc_id": chunk.parent_doc_id,
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "section_path": chunk.section_path,
                        "embedding": vector,
                        "prev_chunk_id": chunk.prev_chunk_id,
                        "next_chunk_id": chunk.next_chunk_id,
                    },
                }
            )

        # Entities. The label is schema-validated, so interpolating it is safe;
        # every value remains a bound parameter.
        for vertex in vertices:
            label = vertex.label
            if not is_safe_identifier(label):
                raise SchemaValidationError(f"Unsafe vertex label reached write stage: {label!r}")
            statements.append(
                {
                    "command": (
                        # `label` is a reserved TinkerPop token and cannot be set as a
                        # property; the vertex type already carries it, and it is
                        # stored as `entity_label` for retrieval-side convenience.
                        f"MERGE (n:{label} {{entity_id: $entity_id}}) "
                        "SET n.name = $name, n.normalized_name = $normalized_name, "
                        "n.aliases = $aliases, n.confidence = $confidence, "
                        "n.mention_count = $mention_count, n.entity_label = $entity_label"
                    ),
                    "params": {
                        "entity_id": vertex.id,
                        "name": vertex.properties.get("name", vertex.id),
                        "normalized_name": vertex.properties.get("normalized_name", ""),
                        "aliases": vertex.properties.get("aliases", []),
                        "confidence": float(vertex.properties.get("confidence", 0.5)),
                        "mention_count": int(vertex.properties.get("mention_count", 1)),
                        "entity_label": label,
                    },
                }
            )

        for edge in edges:
            etype = edge.type
            if not is_safe_identifier(etype):
                raise SchemaValidationError(f"Unsafe edge type reached write stage: {etype!r}")
            statements.append(
                {
                    "command": (
                        "MATCH (a {entity_id: $source}), (b {entity_id: $target}) "
                        f"MERGE (a)-[r:{etype}]->(b) "
                        "SET r.confidence = $confidence, r.chunk_id = $chunk_id, "
                        "r.evidence = $evidence"
                    ),
                    "params": {
                        "source": edge.source,
                        "target": edge.target,
                        "confidence": edge.confidence,
                        "chunk_id": edge.properties.get("chunk_id", ""),
                        "evidence": str(edge.properties.get("evidence", ""))[:500],
                    },
                }
            )

        # MENTIONED_IN bridges the graph back to text, so a traversal can return
        # the passages that support it.
        for entity_id, chunk_id in sorted(mentions):
            statements.append(
                {
                    "command": (
                        "MATCH (e {entity_id: $entity_id}), (c:Chunk {chunk_id: $chunk_id}) "
                        "MERGE (e)-[:MENTIONED_IN]->(c)"
                    ),
                    "params": {"entity_id": entity_id, "chunk_id": chunk_id},
                }
            )

        return await arcadedb_client.execute_batch(statements, tenant_id=tenant_id)

    @staticmethod
    def _empty_result(doc_id: str, tenant_id: str, telemetry: TelemetryTracker) -> Dict[str, Any]:
        result = telemetry.finalize()
        return {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "chunks_created": 0,
            "entities_extracted": 0,
            "relationships_created": 0,
            "mentions_linked": 0,
            "schema_rejections": 0,
            "statements_executed": 0,
            "embedding_model": embedding_service.model_label,
            "extraction_backend": extraction_service.active_backend,
            "telemetry": result,
            "status": "empty_document",
            "execution_time_ms": result["latency_breakdown_ms"]["total_retrieval_latency"],
        }


ingestion_service = IngestionService()
