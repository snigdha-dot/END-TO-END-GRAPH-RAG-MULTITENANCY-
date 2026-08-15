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
        entity_ids = await self._entities_in_chunks(seed_chunk_ids, tenant_id)
        if not entity_ids:
            return Subgraph(), []

        subgraph = await self._walk(entity_ids, tenant_id, schema, hops)
        if not subgraph.nodes:
            return Subgraph(), []

        # Pull chunks the expanded entities appear in, excluding what fusion
        # already returned so expansion adds context rather than repeating it.
        already_seen = set(exclude_chunk_ids or set()) | set(seed_chunk_ids)
        new_chunks = await self._chunks_for_entities(
            [n.id for n in subgraph.nodes], tenant_id, already_seen
        )
        return subgraph, new_chunks

    async def _entities_in_chunks(
        self, chunk_ids: Sequence[str], tenant_id: str
    ) -> List[str]:
        try:
            rows = await arcadedb_client.execute_cypher(
                "MATCH (e)-[:MENTIONED_IN]->(c:Chunk) "
                "WHERE c.chunk_id IN $chunk_ids "
                "RETURN DISTINCT e.entity_id AS entity_id LIMIT $limit",
                {"chunk_ids": list(chunk_ids), "limit": self.MAX_EXPANSION_NODES},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
            )
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            logger.warning("Expansion seed lookup failed (%s); skipping expansion.", exc)
            return []
        return [str(r["entity_id"]) for r in rows if isinstance(r, dict) and r.get("entity_id")]

    async def _walk(
        self, entity_ids: Sequence[str], tenant_id: str, schema, hops: int
    ) -> Subgraph:
        from app.core.security import CypherParameterizer  # noqa: PLC0415

        depth = max(1, min(hops, settings.MAX_TRAVERSAL_DEPTH))
        cypher, params = CypherParameterizer.build_parameterized_traversal(
            start_node_ids=list(entity_ids)[: self.MAX_EXPANSION_NODES],
            max_depth=depth,
            schema=schema,
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

        edges = await self._edges_between([n.id for n in nodes], tenant_id)
        return Subgraph(nodes=nodes, edges=edges)

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
        try:
            rows = await arcadedb_client.execute_cypher(
                "MATCH (e)-[:MENTIONED_IN]->(c:Chunk) "
                "WHERE e.entity_id IN $ids "
                "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, "
                "c.parent_doc_id AS parent_doc_id, c.section_path AS section_path, "
                "c.citation AS citation LIMIT $limit",
                {"ids": list(entity_ids), "limit": settings.MAX_TRAVERSAL_NODES},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
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
        """Rank community reports by similarity to the query."""
        try:
            rows = await arcadedb_client.execute_sql(
                "SELECT chunk_id, text, embedding, community_size, community_rank "
                "FROM Chunk WHERE chunk_kind = 'community_report' LIMIT :limit",
                {"limit": 500},
                tenant_id=tenant_id,
            )
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            logger.warning("Community search failed (%s); no reports returned.", exc)
            return []

        scored: List[Tuple[float, RetrievedChunk]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            similarity = embedding_service.cosine_similarity(
                query_vector, [float(x) for x in embedding]
            )
            if similarity <= 0:
                continue
            # A denser community is a more coherent theme, so its report is more
            # likely to be a real answer than an incidental cluster's.
            rank_weight = 1.0 + min(float(row.get("community_rank") or 0.0), 2.0) * 0.1
            scored.append(
                (
                    similarity * rank_weight,
                    RetrievedChunk(
                        chunk_id=str(row.get("chunk_id", "")),
                        text=str(row.get("text", "")),
                        parent_doc_id="communities",
                        score=round(similarity, 4),
                        retrieval_path="community",
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        results: List[RetrievedChunk] = []
        for rank, (_, chunk) in enumerate(scored[:top_k], start=1):
            chunk.rank = rank
            results.append(chunk)
        return results


graph_expander = GraphExpander()
community_search = CommunitySearch()
