"""Knowledge graph construction: resolved entities and relations into ArcadeDB.

Rebuilt as a single writer for every source format, replacing the two parallel
write paths that had to be fixed in lockstep.

Five behaviours are carried forward deliberately, each learned from running
against a real ArcadeDB rather than from the Cypher specification:

  1. Endpoint matching must name both labels. An untyped
     `MATCH (a {entity_id: ...}), (b {entity_id: ...})` scans every vertex type
     for both endpoints and degrades into a cartesian product: measured at
     65,316ms on a 130-chunk graph versus 1,144ms when labelled, because only the
     labelled form uses the UNIQUE index.
  2. `label` is a reserved TinkerPop token and cannot be a vertex property, so
     the schema label is stored as `entity_label`.
  3. Canonical ids are label-scoped, so the film "Dune" and the studio "Dune"
     remain distinct entities.
  4. Writes are one statement per request with bounded concurrency. ArcadeDB
     24.11.1 has no batch endpoint, and `sqlscript` does not accept bound
     parameters - using it would trade the injection guarantee for throughput.
  5. Every value is a bound parameter. The only interpolated strings are labels
     and edge types already validated against the tenant schema.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.exceptions import SchemaValidationError
from app.core.tenant_schema import TenantGraphSchema, is_safe_identifier
from app.models.canonical import CanonicalChunk
from app.models.graph import Edge, Vertex
from app.services.arcadedb_client import arcadedb_client

logger = logging.getLogger(__name__)


@dataclass
class GraphWriteResult:
    """What reached the database, and what was refused before it got there."""

    chunks_written: int = 0
    entities_written: int = 0
    edges_written: int = 0
    mentions_written: int = 0
    statements_executed: int = 0
    rejected_entities: int = 0
    rejected_edges: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    def note_rejection(self, reason: str) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks_written": self.chunks_written,
            "entities_written": self.entities_written,
            "edges_written": self.edges_written,
            "mentions_written": self.mentions_written,
            "statements_executed": self.statements_executed,
            "rejected_entities": self.rejected_entities,
            "rejected_edges": self.rejected_edges,
            "rejection_reasons": self.rejection_reasons,
        }


class GraphValidator:
    """Last gate before a statement is built. Nothing unvalidated is interpolated."""

    def validate_entities(
        self, entities: Sequence[Vertex], schema: TenantGraphSchema, result: GraphWriteResult
    ) -> List[Vertex]:
        valid: List[Vertex] = []
        for vertex in entities:
            if not schema.validate_vertex_label(vertex.label):
                result.rejected_entities += 1
                result.note_rejection(f"label_not_in_schema:{vertex.label}")
                continue
            if not is_safe_identifier(vertex.label):
                # Should be unreachable: schema labels are validated at load time.
                raise SchemaValidationError(
                    f"Unsafe vertex label reached the write stage: {vertex.label!r}"
                )
            if not vertex.id or not vertex.properties.get("name"):
                result.rejected_entities += 1
                result.note_rejection("missing_id_or_name")
                continue
            valid.append(vertex)
        return valid

    def validate_edges(
        self,
        edges: Sequence[Edge],
        schema: TenantGraphSchema,
        known_ids: Set[str],
        result: GraphWriteResult,
    ) -> List[Edge]:
        valid: List[Edge] = []
        for edge in edges:
            if not schema.validate_edge_type(edge.type):
                result.rejected_edges += 1
                result.note_rejection(f"edge_not_in_schema:{edge.type}")
                continue
            if not is_safe_identifier(edge.type):
                raise SchemaValidationError(
                    f"Unsafe edge type reached the write stage: {edge.type!r}"
                )
            if edge.confidence < settings.EDGE_CONFIDENCE_THRESHOLD:
                result.rejected_edges += 1
                result.note_rejection("below_confidence_threshold")
                continue
            # An edge to an entity that resolution dropped would dangle.
            if edge.source not in known_ids or edge.target not in known_ids:
                result.rejected_edges += 1
                result.note_rejection("dangling_endpoint")
                continue
            if edge.source == edge.target:
                result.rejected_edges += 1
                result.note_rejection("self_loop")
                continue
            valid.append(edge)
        return valid


class GraphBuilder:
    """Writes chunks, entities, relations, and mention links for one tenant."""

    def __init__(self) -> None:
        self.validator = GraphValidator()

    async def write(
        self,
        tenant_id: str,
        schema: TenantGraphSchema,
        chunks: Sequence[CanonicalChunk],
        vectors: Sequence[Sequence[float]],
        entities: Sequence[Vertex],
        edges: Sequence[Edge],
        mentions: Iterable[Tuple[str, str]],
    ) -> GraphWriteResult:
        """Validate then write the whole graph fragment for a batch of chunks."""
        result = GraphWriteResult()

        valid_entities = self.validator.validate_entities(entities, schema, result)
        known_ids = {v.id for v in valid_entities}
        valid_edges = self.validator.validate_edges(edges, schema, known_ids, result)

        label_by_id = {v.id: v.label for v in valid_entities}
        valid_mentions = {
            (entity_id, chunk_id)
            for entity_id, chunk_id in mentions
            if entity_id in known_ids
        }

        statements: List[Dict[str, Any]] = []
        statements.extend(self._chunk_statements(chunks, vectors))
        statements.extend(self._entity_statements(valid_entities))
        statements.extend(self._edge_statements(valid_edges, label_by_id))
        statements.extend(self._mention_statements(valid_mentions, label_by_id))

        if not statements:
            return result

        result.statements_executed = await arcadedb_client.execute_batch(
            statements, tenant_id=tenant_id
        )
        result.chunks_written = len(chunks)
        result.entities_written = len(valid_entities)
        result.edges_written = len(valid_edges)
        result.mentions_written = len(valid_mentions)
        return result

    # ------------------------------------------------------------------ chunks
    @staticmethod
    def _chunk_statements(
        chunks: Sequence[CanonicalChunk], vectors: Sequence[Sequence[float]]
    ) -> List[Dict[str, Any]]:
        """Upsert chunks with their embedding and provenance.

        MERGE rather than CREATE so re-ingesting a document updates it in place
        instead of duplicating every chunk.
        """
        statements: List[Dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            statements.append(
                {
                    "command": (
                        "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                        "SET c.text = $text, c.parent_doc_id = $parent_doc_id, "
                        "c.chunk_kind = $chunk_kind, c.token_count = $token_count, "
                        "c.section_path = $section_path, c.embedding = $embedding, "
                        "c.prev_chunk_id = $prev_chunk_id, c.next_chunk_id = $next_chunk_id, "
                        "c.source_uri = $source_uri, c.source_format = $source_format, "
                        "c.citation = $citation"
                    ),
                    "params": {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "parent_doc_id": chunk.doc_id,
                        "chunk_kind": chunk.kind.value,
                        "token_count": chunk.token_count,
                        "section_path": chunk.section_path,
                        "embedding": list(vector),
                        "prev_chunk_id": chunk.prev_chunk_id,
                        "next_chunk_id": chunk.next_chunk_id,
                        "source_uri": chunk.provenance.source_uri,
                        "source_format": chunk.provenance.source_format,
                        "citation": chunk.provenance.describe(),
                    },
                }
            )
        return statements

    # ---------------------------------------------------------------- entities
    @staticmethod
    def _entity_statements(entities: Sequence[Vertex]) -> List[Dict[str, Any]]:
        statements: List[Dict[str, Any]] = []
        for vertex in entities:
            label = vertex.label  # already schema-validated
            statements.append(
                {
                    "command": (
                        # `label` is a reserved TinkerPop token and cannot be set as
                        # a property; the vertex type already carries it, and it is
                        # mirrored as `entity_label` for the read path.
                        f"MERGE (n:{label} {{entity_id: $entity_id}}) "
                        "SET n.name = $name, n.normalized_name = $normalized_name, "
                        "n.aliases = $aliases, n.confidence = $confidence, "
                        "n.mention_count = $mention_count, n.entity_label = $entity_label, "
                        "n.extractor = $extractor"
                    ),
                    "params": {
                        "entity_id": vertex.id,
                        "name": vertex.properties.get("name", vertex.id),
                        "normalized_name": vertex.properties.get("normalized_name", ""),
                        "aliases": vertex.properties.get("aliases", []),
                        "confidence": float(vertex.properties.get("confidence", 0.5)),
                        "mention_count": int(vertex.properties.get("mention_count", 1)),
                        "entity_label": label,
                        "extractor": str(vertex.properties.get("extractor", "")),
                    },
                }
            )
        return statements

    # ------------------------------------------------------------------- edges
    @staticmethod
    def _edge_statements(
        edges: Sequence[Edge], label_by_id: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Build edge writes with both endpoint labels named.

        Naming both labels is what lets the UNIQUE index on entity_id resolve each
        endpoint. Without them ArcadeDB scans every vertex type for both sides and
        the write degrades into a cartesian product.
        """
        statements: List[Dict[str, Any]] = []
        for edge in edges:
            source_label = label_by_id.get(edge.source)
            target_label = label_by_id.get(edge.target)
            if not source_label or not target_label:
                continue
            statements.append(
                {
                    "command": (
                        f"MATCH (a:{source_label} {{entity_id: $source}}) "
                        f"MATCH (b:{target_label} {{entity_id: $target}}) "
                        f"MERGE (a)-[r:{edge.type}]->(b) "
                        "SET r.confidence = $confidence, r.chunk_id = $chunk_id, "
                        "r.evidence = $evidence, r.extractor = $extractor"
                    ),
                    "params": {
                        "source": edge.source,
                        "target": edge.target,
                        "confidence": edge.confidence,
                        "chunk_id": str(edge.properties.get("chunk_id", "")),
                        "evidence": str(edge.properties.get("evidence", ""))[:500],
                        "extractor": str(edge.properties.get("extractor", "")),
                    },
                }
            )
        return statements

    # ---------------------------------------------------------------- mentions
    @staticmethod
    def _mention_statements(
        mentions: Set[Tuple[str, str]], label_by_id: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """MENTIONED_IN bridges the graph back to text.

        Without it a traversal returns entity names with no passage to cite, and
        the graph path cannot contribute retrievable context.
        """
        statements: List[Dict[str, Any]] = []
        for entity_id, chunk_id in sorted(mentions):
            label = label_by_id.get(entity_id)
            if not label:
                continue
            statements.append(
                {
                    "command": (
                        f"MATCH (e:{label} {{entity_id: $entity_id}}) "
                        "MATCH (c:Chunk {chunk_id: $chunk_id}) "
                        "MERGE (e)-[:MENTIONED_IN]->(c)"
                    ),
                    "params": {"entity_id": entity_id, "chunk_id": chunk_id},
                }
            )
        return statements


graph_builder = GraphBuilder()
