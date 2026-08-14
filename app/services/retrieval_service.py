"""Multi-hop hybrid Graph RAG retrieval (plan section 4).

    Stage 1  Query intent parsing & entity linking
    Stage 2  Dual-path search: vector KNN (Path A) + multi-hop traversal (Path B)
    Stage 3  RRF fusion, cross-encoder reranking, centrality pruning
    Stage 4  Defensive fallback + side-by-side telemetry

The critical change from the previous implementation: seed entities are now
*looked up* in the graph rather than fabricated from query tokens. Previously
"Which films did the director of Inception make?" produced seeds
`canon_which`, `canon_films`, `canon_did` — stopwords matching nothing — so the
graph path could never return a result even against a fully populated database.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from app.core.security import CypherParameterizer
from app.core.telemetry import TelemetryTracker
from app.core.tenant_context import TenantContext
from app.models.graph import Edge, LinkedEntity, RetrievedChunk, Subgraph, Vertex
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.extraction_service import normalize_entity_name
from app.services.reranker_service import reranker_service
from app.services.resolution_service import resolution_service

logger = logging.getLogger(__name__)

# Words that are never entity mentions. Extracting these as seeds is what made
# the previous linker return nothing.
_QUERY_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "done", "doing", "can", "could", "shall", "should",
    "will", "would", "may", "might", "must", "have", "has", "had",
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "that", "this",
    "these", "those", "there", "here", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "about", "into", "over", "after", "before", "also",
    "all", "any", "some", "other", "another", "same", "such", "only", "own",
    "me", "my", "you", "your", "them", "they", "their",
    "show", "tell", "give", "list", "find", "get", "make", "made",
    "things", "stuff", "info", "information",
}

_MENTION_RE = re.compile(r"\b[A-Z][a-zA-Z0-9'’\-]*(?:\s+[A-Z][a-zA-Z0-9'’\-]*)*\b")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


class HybridRetrievalService:
    """Executes the four-stage hybrid retrieval pipeline for one tenant."""

    # ------------------------------------------------- Stage 1: entity linking
    def _candidate_mentions(self, query: str) -> List[str]:
        """Extract plausible entity mentions from the query.

        Prefers capitalized spans (proper nouns); falls back to content words so
        an all-lowercase query still links.
        """
        mentions: List[str] = []
        seen: set[str] = set()

        for match in _MENTION_RE.finditer(query):
            phrase = match.group(0).strip()
            if len(phrase) < 2:
                continue
            words = phrase.split()
            # Strip a leading interrogative: "Which Films" -> "Films".
            while words and words[0].lower() in _QUERY_STOPWORDS:
                words.pop(0)
            while words and words[-1].lower() in _QUERY_STOPWORDS:
                words.pop()
            if not words:
                continue
            cleaned = " ".join(words)
            key = normalize_entity_name(cleaned)
            if key and key not in seen and len(key) > 1:
                seen.add(key)
                mentions.append(cleaned)

        if not mentions:
            for word in _WORD_RE.findall(query):
                lowered = word.lower()
                if lowered in _QUERY_STOPWORDS or len(lowered) < 3:
                    continue
                key = normalize_entity_name(word)
                if key and key not in seen:
                    seen.add(key)
                    mentions.append(word)

        return mentions[:8]

    async def _link_entities(
        self, query: str, tenant_id: str
    ) -> Tuple[List[LinkedEntity], List[str]]:
        """Resolve query mentions to canonical graph entities via DB lookup."""
        mentions = self._candidate_mentions(query)
        if not mentions:
            return [], []

        # Look up all mention surface forms and their sub-phrases in one query.
        lookup_names: List[str] = []
        for mention in mentions:
            normalized = normalize_entity_name(mention)
            if normalized:
                lookup_names.append(normalized)
            for word in mention.split():
                sub = normalize_entity_name(word)
                if sub and len(sub) > 2 and sub not in lookup_names:
                    lookup_names.append(sub)

        cypher, params = CypherParameterizer.build_entity_candidate_lookup(lookup_names, limit=60)
        rows = await arcadedb_client.execute_cypher(cypher, params, tenant_id=tenant_id)

        candidates: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            entity_id = row.get("entity_id")
            if not entity_id:
                continue
            candidates.append(
                {
                    "entity_id": str(entity_id),
                    "name": str(row.get("name", entity_id)),
                    "label": str(row.get("label", "Entity")),
                    "normalized_name": str(row.get("normalized_name", "")),
                    "aliases": row.get("aliases") or [],
                }
            )

        if not candidates:
            return [], mentions

        linked: List[LinkedEntity] = []
        seen_ids: set[str] = set()
        mention_vectors = embedding_service.encode_batch(mentions)

        for mention, vector in zip(mentions, mention_vectors):
            match = resolution_service.link_mention_to_candidates(mention, candidates, vector)
            if match is None:
                continue
            entity_id = match["entity_id"]
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            linked.append(
                LinkedEntity(
                    mention=mention,
                    entity_id=entity_id,
                    name=match["name"],
                    label=match["label"],
                    score=float(match["score"]),
                    method=str(match["method"]),
                )
            )

        return linked, mentions

    # ------------------------------------------------- Stage 2A: vector search
    async def _vector_search(
        self, query_vector: List[float], tenant_id: str, top_k: int
    ) -> List[RetrievedChunk]:
        """Path A: dense KNN over the HNSW chunk index.

        Tries the native vector function first; falls back to scoring candidate
        chunks in-process when the ArcadeDB build lacks HNSW support.
        """
        knn_limit = max(top_k * 3, 15)
        try:
            rows = await arcadedb_client.execute_sql(
                "SELECT chunk_id, text, parent_doc_id, section_path, "
                "vectorNeighbors('Chunk[embedding]', :vec, :k) AS neighbors "
                "FROM Chunk LIMIT 1",
                {"vec": query_vector, "k": knn_limit},
                tenant_id=tenant_id,
            )
            chunks = self._parse_knn_rows(rows, top_k)
            if chunks:
                return chunks
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            if isinstance(exc, DatabaseConnectionError):
                raise
            logger.debug("Native KNN unavailable (%s); scoring chunks in-process.", exc.detail)

        return await self._vector_search_fallback(query_vector, tenant_id, top_k)

    def _parse_knn_rows(self, rows: List[Dict[str, Any]], top_k: int) -> List[RetrievedChunk]:
        chunks: List[RetrievedChunk] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            neighbors = row.get("neighbors")
            if not isinstance(neighbors, list):
                continue
            for rank, neighbor in enumerate(neighbors[:top_k], start=1):
                if not isinstance(neighbor, dict):
                    continue
                vertex = neighbor.get("vertex") or neighbor
                chunk_id = vertex.get("chunk_id")
                if not chunk_id:
                    continue
                distance = float(neighbor.get("distance", 0.0))
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_id),
                        text=str(vertex.get("text", "")),
                        parent_doc_id=str(vertex.get("parent_doc_id", "")),
                        score=round(max(0.0, 1.0 - distance), 4),
                        section_path=list(vertex.get("section_path") or []),
                        retrieval_path="vector",
                        rank=rank,
                    )
                )
        return chunks

    async def _vector_search_fallback(
        self, query_vector: List[float], tenant_id: str, top_k: int
    ) -> List[RetrievedChunk]:
        """Cosine-score chunks in-process. Bounded so it cannot scan a huge KB."""
        rows = await arcadedb_client.execute_sql(
            "SELECT chunk_id, text, parent_doc_id, section_path, embedding "
            "FROM Chunk LIMIT :limit",
            {"limit": settings.MAX_TRAVERSAL_NODES * 5},
            tenant_id=tenant_id,
        )

        scored: List[Tuple[float, RetrievedChunk]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            embedding = row.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue
            score = embedding_service.cosine_similarity(query_vector, [float(x) for x in embedding])
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    RetrievedChunk(
                        chunk_id=str(row.get("chunk_id", "")),
                        text=str(row.get("text", "")),
                        parent_doc_id=str(row.get("parent_doc_id", "")),
                        score=round(score, 4),
                        section_path=list(row.get("section_path") or []),
                        retrieval_path="vector",
                    ),
                )
            )

        scored.sort(key=lambda t: t[0], reverse=True)
        out: List[RetrievedChunk] = []
        for rank, (_, chunk) in enumerate(scored[:top_k], start=1):
            chunk.rank = rank
            out.append(chunk)
        return out

    # ------------------------------------------------- Stage 2B: graph traversal
    async def _graph_traversal(
        self, seed_ids: List[str], tenant_id: str, ctx: TenantContext, max_depth: int
    ) -> Subgraph:
        """Path B: bounded multi-hop traversal from linked seed entities."""
        if not seed_ids:
            return Subgraph()

        cypher, params = CypherParameterizer.build_parameterized_traversal(
            start_node_ids=seed_ids, max_depth=max_depth, schema=ctx.schema
        )
        rows = await arcadedb_client.execute_cypher(cypher, params, tenant_id=tenant_id)
        return self._parse_traversal(rows)

    def _parse_traversal(self, rows: List[Dict[str, Any]]) -> Subgraph:
        nodes: List[Vertex] = []
        edges: List[Edge] = []
        seen_nodes: set[str] = set()
        seen_edges: set[Tuple[str, str, str]] = set()

        for row in rows:
            if not isinstance(row, dict):
                continue

            for node in row.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                node_id = node.get("entity_id") or node.get("id") or node.get("@rid")
                if not node_id or str(node_id) in seen_nodes:
                    continue
                seen_nodes.add(str(node_id))
                properties = {k: v for k, v in node.items() if k != "embedding"}
                nodes.append(
                    Vertex(
                        id=str(node_id),
                        label=str(node.get("@type") or node.get("@class") or node.get("label") or "Entity"),
                        properties=properties,
                    )
                )

            for edge in row.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                source = edge.get("source") or edge.get("out") or edge.get("@out")
                target = edge.get("target") or edge.get("in") or edge.get("@in")
                etype = str(edge.get("@type") or edge.get("@class") or edge.get("type") or "RELATED_TO")
                if not source or not target:
                    continue
                key = (str(source), etype, str(target))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    Edge(
                        source=str(source),
                        target=str(target),
                        type=etype,
                        properties={k: v for k, v in edge.items() if k != "embedding"},
                    )
                )

        return Subgraph(nodes=nodes, edges=edges)

    async def _chunks_for_entities(
        self, entity_ids: Sequence[str], tenant_id: str, limit: int
    ) -> List[RetrievedChunk]:
        """Fetch chunks that mention the traversed entities (graph -> text bridge)."""
        if not entity_ids:
            return []

        rows = await arcadedb_client.execute_cypher(
            "MATCH (e)-[:MENTIONED_IN]->(c:Chunk) "
            "WHERE e.entity_id IN $entity_ids "
            "RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, "
            "c.parent_doc_id AS parent_doc_id, c.section_path AS section_path "
            "LIMIT $limit",
            {"entity_ids": list(entity_ids), "limit": int(limit)},
            tenant_id=tenant_id,
        )

        out: List[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or not row.get("chunk_id"):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    text=str(row.get("text", "")),
                    parent_doc_id=str(row.get("parent_doc_id", "")),
                    section_path=list(row.get("section_path") or []),
                    retrieval_path="graph",
                    rank=rank,
                )
            )
        return out

    # ------------------------------------------------- passage construction
    @staticmethod
    def _subgraph_passages(subgraph: Subgraph, seed_ids: Sequence[str], limit: int) -> List[str]:
        """Verbalize graph relationships so an LLM can read them as context.

        This is what carries multi-hop answers: the fact linking two entities lives
        in the edge, not in any single chunk of text.
        """
        by_id = {n.id: n for n in subgraph.nodes}
        passages: List[str] = []
        seeds = set(seed_ids)

        # Edges touching a seed first: those answer the question most directly.
        ordered = sorted(
            subgraph.edges,
            key=lambda e: (e.source in seeds or e.target in seeds, e.confidence),
            reverse=True,
        )
        for edge in ordered[:limit]:
            source = by_id.get(edge.source)
            target = by_id.get(edge.target)
            if source is None or target is None:
                continue
            relation = edge.type.replace("_", " ").lower()
            passages.append(f"{source.name} ({source.label}) {relation} {target.name} ({target.label}).")
        return passages

    # ------------------------------------------------- Stage 1-4 orchestration
    async def execute_retrieval(
        self,
        ctx: TenantContext,
        query: str,
        max_depth: int = 2,
        top_k: int = 5,
        include_vector_search: bool = True,
        disable_graph_path: bool = False,
    ) -> Dict[str, Any]:
        """Run the full pipeline and return subgraph, passages, and telemetry.

        `disable_graph_path` suppresses Path B, reducing the pipeline to vector-only
        retrieval. Used by the evaluation harness as an ablation control: comparing
        the two isolates what multi-hop traversal actually contributes.
        """
        telemetry = TelemetryTracker()
        tenant_id = ctx.tenant_id
        safe_query = CypherParameterizer.guard_user_text(query, "user_query")

        # ---------------------------------------------------------- Stage 1
        t0 = time.perf_counter()
        query_vector = embedding_service.encode_query(safe_query)
        linked, mentions = await self._link_entities(safe_query, tenant_id)
        seed_ids = [entity.entity_id for entity in linked]
        stage1_ms = (time.perf_counter() - t0) * 1000

        telemetry.record_step_latency("query_entity_linking", stage1_ms)
        telemetry.record_model_call(
            step_name="query_entity_linking",
            model_name=embedding_service.model_label,
            prompt_tokens=len(safe_query.split()),
            completion_tokens=0,
            duration_ms=stage1_ms,
        )

        # ---------------------------------------------------------- Stage 2
        vector_chunks: List[RetrievedChunk] = []
        graph_subgraph = Subgraph()
        graph_chunks: List[RetrievedChunk] = []

        t_vec = time.perf_counter()
        if include_vector_search:
            vector_chunks = await self._vector_search(query_vector, tenant_id, top_k * 2)
        vector_ms = (time.perf_counter() - t_vec) * 1000
        # Always emit this key: plan section 5 declares it as part of the contract.
        telemetry.record_step_latency("arcadedb_vector_knn", vector_ms)

        t_graph = time.perf_counter()
        if seed_ids and not disable_graph_path:
            graph_subgraph = await self._graph_traversal(seed_ids, tenant_id, ctx, max_depth)
            entity_ids = [n.id for n in graph_subgraph.nodes]
            graph_chunks = await self._chunks_for_entities(entity_ids, tenant_id, top_k * 2)
        graph_ms = (time.perf_counter() - t_graph) * 1000
        telemetry.record_step_latency("arcadedb_cypher_traversal", graph_ms)

        # ---------------------------------------------------------- Stage 3
        t_rank = time.perf_counter()
        fused = reranker_service.fuse_paths(safe_query, vector_chunks, graph_chunks, top_k)
        if graph_subgraph.nodes:
            graph_subgraph = reranker_service.prune_subgraph_by_centrality(
                graph_subgraph, seed_ids, settings.MAX_TRAVERSAL_NODES
            )
        rank_ms = (time.perf_counter() - t_rank) * 1000
        telemetry.record_step_latency("rrf_reranking", rank_ms)
        if fused:
            telemetry.record_model_call(
                step_name="rrf_reranking",
                model_name=reranker_service.model_label,
                prompt_tokens=sum(len(c.text.split()) for c in fused),
                completion_tokens=0,
                duration_ms=rank_ms,
            )

        # ---------------------------------------------------------- Stage 4
        passages: List[str] = []
        graph_passages = self._subgraph_passages(graph_subgraph, seed_ids, top_k)
        passages.extend(graph_passages)
        passages.extend(chunk.contextual_text() if hasattr(chunk, "contextual_text") else chunk.text
                        for chunk in fused)

        fallback_used = False
        if not passages:
            # Genuine fallback: a broader pure-vector sweep, not a synthetic string.
            fallback_used = True
            t_fb = time.perf_counter()
            broader = await self._vector_search(query_vector, tenant_id, max(top_k * 2, 10))
            fallback_ms = (time.perf_counter() - t_fb) * 1000
            telemetry.record_step_latency("defensive_vector_fallback", fallback_ms)
            reranked = reranker_service.rerank_chunks(safe_query, broader, top_k)
            passages = [c.text for c in reranked]
            fused = reranked

        telemetry_output = telemetry.finalize()
        telemetry_output["retrieval_diagnostics"] = {
            "linked_entities": [e.model_dump() for e in linked],
            "candidate_mentions": mentions,
            "seed_entity_count": len(seed_ids),
            "vector_hits": len(vector_chunks),
            "graph_nodes": len(graph_subgraph.nodes),
            "graph_edges": len(graph_subgraph.edges),
            "graph_chunk_hits": len(graph_chunks),
            "fallback_used": fallback_used,
            "graph_path_disabled": disable_graph_path,
            "semantic_embeddings": embedding_service.is_semantic,
            "cross_encoder_active": reranker_service.has_cross_encoder,
        }

        return {
            "subgraph": graph_subgraph,
            "passages": passages[: max(top_k * 2, top_k)],
            "chunks": fused,
            "linked_entities": linked,
            "telemetry": telemetry_output,
        }


retrieval_service = HybridRetrievalService()
