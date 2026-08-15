"""Graph expansion and community search — the two post-fusion stages.

GraphExpander runs *after* rank fusion, on entities found in the top-ranked
chunks rather than on the query's seeds. That difference is the point: initial
traversal starts from what the query named, while expansion starts from what
retrieval actually surfaced, which is often a different and better set. A query
mentioning one drug can rank a chunk about a related condition, and expanding
from that condition reaches material the seed never would.

CommunitySearch serves global queries. Reports were stored as Chunk vertices at
ingestion, so they are searchable through the same vector path as ordinary
content, with no second retrieval mechanism to maintain.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from app.models.graph import Edge, RetrievedChunk, Subgraph, Vertex
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


class GraphExpander:
    """Expands the retrieved set by 1-2 hops from entities in top-ranked chunks."""

    MAX_SEED_CHUNKS = 8
    MAX_EXPANSION_NODES = 60

    async def expand(
        self,
        chunks: Sequence[RetrievedChunk],
        tenant_id: str,
        schema,
        hops: int = 1,
        exclude_chunk_ids: Optional[Set[str]] = None,
    ) -> Tuple[Subgraph, List[RetrievedChunk]]:
        """Find entities in the top chunks, walk outward, return new context."""
        if not chunks or hops <= 0:
            return Subgraph(), []

        seed_chunk_ids = [c.chunk_id for c in chunks[: self.MAX_SEED_CHUNKS]]
        entities = await self._entities_in_chunks(seed_chunk_ids, tenant_id)
        if not entities:
            return Subgraph(), []

        # Group by label so each traversal can name its start label. An untyped
        # start scans every vertex type instead of using the UNIQUE index on
        # entity_id: 61,901ms versus 35ms on a 400-chunk tenant.
        by_label: Dict[str, List[str]] = {}
        for entity_id, label in entities:
            by_label.setdefault(label or "", []).append(entity_id)

        nodes: List[Vertex] = []
        seen: Set[str] = set()
        for label, ids in by_label.items():
            walked = await self._walk(ids, tenant_id, schema, hops, label or None)
            for node in walked.nodes:
                if node.id not in seen:
                    seen.add(node.id)
                    nodes.append(node)

        if not nodes:
            return Subgraph(), []

        edges = await self._edges_between([n.id for n in nodes], tenant_id)
        subgraph = Subgraph(nodes=nodes, edges=edges)

        # Pull chunks the expanded entities appear in, excluding what fusion
        # already returned so expansion adds context rather than repeating it.
        already_seen = set(exclude_chunk_ids or set()) | set(seed_chunk_ids)
        new_chunks = await self._chunks_for_entities(
            [n.id for n in subgraph.nodes], tenant_id, already_seen
        )
        return subgraph, new_chunks

    async def _entities_in_chunks(
        self, chunk_ids: Sequence[str], tenant_id: str
    ) -> List[Tuple[str, str]]:
        """Entities mentioned in the given chunks, as (entity_id, label) pairs.

        The label travels with the id because the traversal that follows needs it
        to use the index rather than scan.
        """
        try:
            rows = await arcadedb_client.execute_cypher(
                "MATCH (c:Chunk)<-[:MENTIONED_IN]-(e) "
                "WHERE c.chunk_id IN $chunk_ids "
                "RETURN DISTINCT e.entity_id AS entity_id, "
                "e.entity_label AS entity_label LIMIT $limit",
                {"chunk_ids": list(chunk_ids), "limit": self.MAX_EXPANSION_NODES},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
                result_limit=self.MAX_EXPANSION_NODES,
            )
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            logger.warning("Expansion seed lookup failed (%s); skipping expansion.", exc)
            return []
        return [
            (str(r["entity_id"]), str(r.get("entity_label") or ""))
            for r in rows
            if isinstance(r, dict) and r.get("entity_id")
        ]

    async def _walk(
        self,
        entity_ids: Sequence[str],
        tenant_id: str,
        schema,
        hops: int,
        seed_label: Optional[str] = None,
    ) -> Subgraph:
        from app.core.security import CypherParameterizer  # noqa: PLC0415

        depth = max(1, min(hops, settings.MAX_TRAVERSAL_DEPTH))
        cypher, params = CypherParameterizer.build_parameterized_traversal(
            start_node_ids=list(entity_ids)[: self.MAX_EXPANSION_NODES],
            max_depth=depth,
            schema=schema,
            seed_label=seed_label,
        )
        try:
            rows = await arcadedb_client.execute_cypher(
                cypher, params, tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
            )
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            # Expansion is an enhancement; a slow walk should cost context, not
            # the response.
            logger.warning("Graph expansion walk failed (%s); returning empty.", exc)
            return Subgraph()

        nodes: List[Vertex] = []
        seen: Set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            for prefix in ("source", "target"):
                node_id = row.get(f"{prefix}_id")
                if not node_id or str(node_id) in seen:
                    continue
                seen.add(str(node_id))
                nodes.append(
                    Vertex(
                        id=str(node_id),
                        label=str(row.get(f"{prefix}_label") or "Entity"),
                        properties={
                            "entity_id": str(node_id),
                            "name": str(row.get(f"{prefix}_name") or node_id),
                            "via": "expansion",
                        },
                    )
                )

        # Edges are recovered once by the caller across all label groups, so a
        # per-group fetch here would duplicate the work.
        return Subgraph(nodes=nodes, edges=[])

    async def _edges_between(self, node_ids: Sequence[str], tenant_id: str) -> List[Edge]:
        if not node_ids:
            return []
        try:
            rows = await arcadedb_client.execute_cypher(
                "MATCH (a)-[r]->(b) "
                "WHERE a.entity_id IN $ids AND b.entity_id IN $ids "
                "RETURN a.entity_id AS source_id, type(r) AS rel_type, "
                "b.entity_id AS target_id, r.confidence AS confidence "
                "LIMIT $limit",
                {"ids": list(node_ids), "limit": settings.MAX_TRAVERSAL_NODES},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
            )
        except (DatabaseQueryError, DatabaseConnectionError):
            return []

        edges: List[Edge] = []
        seen: Set[Tuple[str, str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            source, target = row.get("source_id"), row.get("target_id")
            rel_type = str(row.get("rel_type") or "RELATED_TO")
            if not source or not target:
                continue
            key = (str(source), rel_type, str(target))
            if key in seen:
                continue
            seen.add(key)
            confidence = row.get("confidence")
            edges.append(
                Edge(
                    source=str(source),
                    target=str(target),
                    type=rel_type,
                    properties={
                        "confidence": float(confidence) if confidence is not None else 1.0,
                        "via": "expansion",
                    },
                )
            )
        return edges

    async def _chunks_for_entities(
        self, entity_ids: Sequence[str], tenant_id: str, exclude: Set[str]
    ) -> List[RetrievedChunk]:
        if not entity_ids:
            return []
        # Bound both the entities queried and the chunks returned. A traversal can
        # reach 100 entities, and pulling full text for every chunk each of them
        # appears in measured ~4s on a 400-chunk tenant - most of it transferring
        # 34-column records that fusion then discards below its own top-k.
        bounded_ids = list(entity_ids)[: settings.GRAPH_CHUNK_SEED_LIMIT]
        try:
            rows = await arcadedb_client.execute_cypher(
                "MATCH (e)-[:MENTIONED_IN]->(c:Chunk) "
                "WHERE e.entity_id IN $ids "
                "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, "
                "c.parent_doc_id AS parent_doc_id, c.section_path AS section_path, "
                "c.citation AS citation LIMIT $limit",
                {"ids": bounded_ids, "limit": settings.GRAPH_CHUNK_LIMIT},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
                result_limit=settings.GRAPH_CHUNK_LIMIT,
            )
        except (DatabaseQueryError, DatabaseConnectionError):
            return []

        results: List[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            chunk_id = row.get("chunk_id")
            if not chunk_id or str(chunk_id) in exclude:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(row.get("text", "")),
                    parent_doc_id=str(row.get("parent_doc_id", "")),
                    section_path=list(row.get("section_path") or []),
                    retrieval_path="expansion",
                    rank=rank,
                )
            )
        return results


class CommunitySearch:
    """Retrieves community reports for queries with no entity anchor."""

    async def search(
        self, query_vector: Sequence[float], tenant_id: str, top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Rank community reports by similarity to the query.

        Served from the same cached vector index as ordinary retrieval, so a
        global query costs no database round-trip either.
        """
        from app.services.vector_index import vector_index_service  # noqa: PLC0415

        try:
            return await vector_index_service.search_community_reports(
                query_vector, tenant_id, top_k
            )
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            logger.warning("Community search failed (%s); no reports returned.", exc)
            return []


graph_expander = GraphExpander()
community_search = CommunitySearch()
