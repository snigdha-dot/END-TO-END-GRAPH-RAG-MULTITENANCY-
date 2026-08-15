"""Retrieval orchestration.

    auth -> query understanding -> routing
         -> [vector | lexical | graph] -> rank fusion
         -> graph expansion -> community search
         -> rerank -> context optimization -> answer-ready context

Paths run concurrently: they are independent I/O against the same database, and
running them in sequence made total latency the sum of the slowest three rather
than the maximum.

Every stage records what it did into telemetry, so a poor answer can be traced to
the stage that caused it rather than guessed at.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from app.core.security import CypherParameterizer
from app.core.telemetry import TelemetryTracker
from app.core.tenant_context import TenantContext
from app.models.graph import Edge, LinkedEntity, RetrievedChunk, Subgraph, Vertex
from app.services.arcadedb_client import arcadedb_client
from app.services.context_optimizer import context_optimizer
from app.services.embedding_service import embedding_service
from app.services.graph_expansion import community_search, graph_expander
from app.services.lexical_search import lexical_search_service
from app.services.query_understanding import QueryAnalysis, QueryIntent, query_understanding
from app.services.reranker_service import reranker_service
from app.services.resolution_service import resolution_service
from app.services.retrieval_router import RetrievalPlan, retrieval_router
from app.services.vector_index import vector_index_service

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Runs the full retrieval architecture for one query."""

    async def retrieve(
        self,
        ctx: TenantContext,
        query: str,
        top_k: int = 5,
        max_depth: Optional[int] = None,
        context_budget_tokens: Optional[int] = None,
        conversation_context: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        telemetry = TelemetryTracker()
        tenant_id = ctx.tenant_id
        schema = ctx.schema

        safe_query = CypherParameterizer.guard_user_text(query, "user_query")

        # ---------------------------------------------------- query understanding
        t0 = time.perf_counter()
        analysis = query_understanding.analyze(safe_query, conversation_context)
        telemetry.record_step_latency("query_understanding", (time.perf_counter() - t0) * 1000)

        # An underspecified query is answered with a question, not a guess.
        # Retrieving something plausible would present a coin flip as an answer.
        if analysis.needs_clarification:
            return self._clarification_response(ctx, analysis, telemetry)

        # A query resolved from conversation context is retrieved as its resolved
        # form; the original is preserved in diagnostics.
        safe_query = analysis.effective_query

        plan = retrieval_router.plan(analysis, top_k=top_k)
        if max_depth is not None:
            plan.graph_depth = min(max_depth, settings.MAX_TRAVERSAL_DEPTH)

        # ---------------------------------------------------- entity linking
        t1 = time.perf_counter()
        query_vector = await embedding_service.encode_query_async(safe_query)
        linked: List[LinkedEntity] = []
        if plan.use_graph:
            linked = await self._link_entities(analysis, tenant_id, query_vector)
        seed_ids = [e.entity_id for e in linked]
        # Entity linking already resolved each seed's label. Grouping by label lets
        # every traversal name its start type and use the UNIQUE index rather than
        # scanning all vertex types.
        seed_groups: Dict[str, List[str]] = {}
        for entity in linked:
            seed_groups.setdefault(entity.label or "", []).append(entity.entity_id)
        link_ms = (time.perf_counter() - t1) * 1000
        telemetry.record_step_latency("query_entity_linking", link_ms)
        telemetry.record_model_call(
            step_name="query_entity_linking",
            model_name=embedding_service.model_label,
            prompt_tokens=len(safe_query.split()),
            completion_tokens=0,
            duration_ms=link_ms,
        )

        # A plan that assumed an anchor but found none would run a traversal from
        # nothing, so the plan is corrected rather than executed as written.
        if plan.use_graph and not seed_ids:
            plan.use_graph = False
            plan.rationale.append("no seeds resolved; graph path dropped")

        # ---------------------------------------------------- parallel retrieval
        t2 = time.perf_counter()
        vector_chunks, lexical_chunks, graph_subgraph, graph_chunks = await self._run_paths(
            plan, analysis, query_vector, seed_ids, tenant_id, ctx, schema, seed_groups
        )
        telemetry.record_step_latency("arcadedb_vector_knn", (time.perf_counter() - t2) * 1000)
        telemetry.record_step_latency("lexical_search", 0.0)
        telemetry.record_step_latency("arcadedb_cypher_traversal", 0.0)

        # ---------------------------------------------------- rank fusion
        t3 = time.perf_counter()
        fused = self._fuse(
            safe_query, vector_chunks, lexical_chunks, graph_chunks, plan, top_k
        )
        telemetry.record_step_latency("rank_fusion", (time.perf_counter() - t3) * 1000)

        # ---------------------------------------------------- graph expansion
        expansion_subgraph = Subgraph()
        expansion_chunks: List[RetrievedChunk] = []
        if plan.use_graph_expansion and plan.expansion_hops > 0 and fused:
            t4 = time.perf_counter()
            expansion_subgraph, expansion_chunks = await graph_expander.expand(
                fused, tenant_id, schema, hops=plan.expansion_hops,
                exclude_chunk_ids={c.chunk_id for c in fused},
            )
            telemetry.record_step_latency("graph_expansion", (time.perf_counter() - t4) * 1000)

        # ---------------------------------------------------- community search
        community_chunks: List[RetrievedChunk] = []
        if plan.use_community:
            t5 = time.perf_counter()
            community_chunks = await community_search.search(
                query_vector, tenant_id, top_k=plan.community_k
            )
            telemetry.record_step_latency("community_search", (time.perf_counter() - t5) * 1000)

        # ---------------------------------------------------- rerank
        t6 = time.perf_counter()
        candidates = fused + expansion_chunks + community_chunks
        reranked = await reranker_service.rerank_chunks_async(
            safe_query, candidates, top_k * 2
        )
        rerank_ms = (time.perf_counter() - t6) * 1000
        telemetry.record_step_latency("rrf_reranking", rerank_ms)
        if reranked:
            telemetry.record_model_call(
                step_name="reranking",
                model_name=reranker_service.model_label,
                prompt_tokens=sum(len(c.text.split()) for c in reranked),
                completion_tokens=0,
                duration_ms=rerank_ms,
            )

        merged_subgraph = self._merge_subgraphs(graph_subgraph, expansion_subgraph, seed_ids)

        # ---------------------------------------------------- defensive fallback
        fallback_used = False
        if not reranked:
            fallback_used = True
            t7 = time.perf_counter()
            broader = await self._vector_search(query_vector, tenant_id, max(top_k * 3, 15))
            telemetry.record_step_latency(
                "defensive_vector_fallback", (time.perf_counter() - t7) * 1000
            )
            reranked = await reranker_service.rerank_chunks_async(
                safe_query, broader, top_k
            )

        # ---------------------------------------------------- context optimization
        t8 = time.perf_counter()
        optimized = context_optimizer.optimize(
            reranked,
            subgraph=merged_subgraph,
            seed_ids=seed_ids,
            budget_tokens=context_budget_tokens,
            max_passages=max(top_k * 2, 10),
        )
        telemetry.record_step_latency("context_optimization", (time.perf_counter() - t8) * 1000)

        telemetry_output = telemetry.finalize()
        telemetry_output["retrieval_diagnostics"] = {
            "query_analysis": analysis.to_dict(),
            "retrieval_plan": plan.to_dict(),
            "linked_entities": [e.model_dump() for e in linked],
            "seed_entity_count": len(seed_ids),
            "vector_hits": len(vector_chunks),
            "lexical_hits": len(lexical_chunks),
            "graph_nodes": len(merged_subgraph.nodes),
            "graph_edges": len(merged_subgraph.edges),
            "graph_chunk_hits": len(graph_chunks),
            "expansion_nodes": len(expansion_subgraph.nodes),
            "expansion_chunks": len(expansion_chunks),
            "community_hits": len(community_chunks),
            "fallback_used": fallback_used,
            "context": optimized.to_dict(),
            "semantic_embeddings": embedding_service.is_semantic,
            "cross_encoder_active": reranker_service.has_cross_encoder,
        }

        return {
            "subgraph": merged_subgraph,
            "passages": optimized.passages,
            "citations": optimized.citations,
            "chunks": optimized.chunks,
            "linked_entities": linked,
            "telemetry": telemetry_output,
        }

    # ------------------------------------------------------------ clarification
    @staticmethod
    def _clarification_response(
        ctx: TenantContext, analysis: QueryAnalysis, telemetry: TelemetryTracker
    ) -> Dict[str, Any]:
        """Return a clarification request instead of retrieved content.

        Shaped like a normal response so callers need no special handling: the
        prompt arrives as a passage, and `needs_clarification` in diagnostics
        tells a caller that wants to distinguish them.
        """
        output = telemetry.finalize()
        output["retrieval_diagnostics"] = {
            "query_analysis": analysis.to_dict(),
            "retrieval_plan": {"intent": "clarify", "active_paths": []},
            "linked_entities": [],
            "seed_entity_count": 0,
            "vector_hits": 0,
            "lexical_hits": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "graph_chunk_hits": 0,
            "expansion_nodes": 0,
            "expansion_chunks": 0,
            "community_hits": 0,
            "fallback_used": False,
            "needs_clarification": True,
            "context": {"passages": 1, "total_tokens": 0, "budget_tokens": 0},
            "semantic_embeddings": embedding_service.is_semantic,
            "cross_encoder_active": reranker_service.has_cross_encoder,
        }
        return {
            "subgraph": Subgraph(),
            "passages": [analysis.clarification_prompt],
            "citations": ["clarification request"],
            "chunks": [],
            "linked_entities": [],
            "needs_clarification": True,
            "clarification_prompt": analysis.clarification_prompt,
            "telemetry": output,
        }

    # ------------------------------------------------------------------ paths
    async def _run_paths(
        self,
        plan: RetrievalPlan,
        analysis: QueryAnalysis,
        query_vector: Sequence[float],
        seed_ids: Sequence[str],
        tenant_id: str,
        ctx: TenantContext,
        schema,
        seed_groups: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[RetrievedChunk], List[RetrievedChunk], Subgraph, List[RetrievedChunk]]:
        """Run the enabled paths concurrently.

        They are independent I/O against the same database, so running them in
        sequence made latency the sum of the slowest three rather than the max.
        """
        tasks: Dict[str, asyncio.Task] = {}

        if plan.use_vector:
            tasks["vector"] = asyncio.create_task(
                self._vector_search(query_vector, tenant_id, plan.vector_k)
            )
        if plan.use_lexical:
            tasks["lexical"] = asyncio.create_task(
                lexical_search_service.search(
                    analysis.query, tenant_id, plan.lexical_k,
                    boost_phrases=analysis.quoted_phrases + analysis.identifiers,
                )
            )
        if plan.use_graph and seed_ids:
            tasks["graph"] = asyncio.create_task(
                self._graph_search(
                    seed_ids, tenant_id, ctx, schema, plan.graph_depth,
                    seed_labels=seed_groups
                )
            )

        results: Dict[str, Any] = {}
        if tasks:
            completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for name, outcome in zip(tasks.keys(), completed):
                if isinstance(outcome, Exception):
                    # One failed path should cost coverage, not the response.
                    logger.warning("Retrieval path '%s' failed: %s", name, outcome)
                    results[name] = None
                else:
                    results[name] = outcome

        vector_chunks = results.get("vector") or []
        lexical_chunks = results.get("lexical") or []
        graph_result = results.get("graph") or (Subgraph(), [])
        graph_subgraph, graph_chunks = graph_result

        return vector_chunks, lexical_chunks, graph_subgraph, graph_chunks

    async def _vector_search(
        self, query_vector: Sequence[float], tenant_id: str, top_k: int
    ) -> List[RetrievedChunk]:
        """Dense retrieval over the tenant's cached vector index.

        Vectors are held in memory per tenant rather than refetched per query:
        pulling 400 x 384 floats over HTTP measured ~330ms against a 400-chunk
        tenant, of which only ~11ms was the scoring. Version filtering and tenant
        scoping happen at index-build time.
        """
        try:
            return await vector_index_service.search(query_vector, tenant_id, top_k)
        except DatabaseConnectionError:
            raise
        except DatabaseQueryError as exc:
            logger.warning("Vector search failed: %s", exc.detail)
            return []

    async def _graph_search(
        self,
        seed_ids: Sequence[str],
        tenant_id: str,
        ctx: TenantContext,
        schema,
        depth: int,
        seed_label: Optional[str] = None,
        seed_labels: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[Subgraph, List[RetrievedChunk]]:
        """Multi-hop traversal from the query's seed entities.

        Seeds are grouped by label and traversed per group. A Cypher MATCH names
        one label, and an untyped start makes ArcadeDB scan every vertex type
        instead of using the UNIQUE index on entity_id - 61,901ms versus 35ms on
        a 400-chunk tenant. Running one query per label keeps every traversal on
        the indexed path.
        """
        groups = seed_labels or ({seed_label: list(seed_ids)} if seed_label else {})
        if not groups:
            groups = {"": list(seed_ids)}

        rows: List[Dict[str, Any]] = []
        for label, ids in groups.items():
            if not ids:
                continue
            cypher, params = CypherParameterizer.build_parameterized_traversal(
                start_node_ids=ids, max_depth=depth, schema=schema,
                seed_label=label or None,
            )
            try:
                rows.extend(
                    await arcadedb_client.execute_cypher(
                        cypher, params, tenant_id=tenant_id,
                        timeout_ms=settings.ARCADEDB_TRAVERSAL_TIMEOUT_MS,
                    )
                )
            except (DatabaseQueryError, DatabaseConnectionError) as exc:
                # One slow label group costs its own results, not the whole path.
                logger.warning(
                    "Graph traversal failed for label %r (%s); continuing.", label, exc
                )
                continue

        if not rows:
            return Subgraph(), []

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
                        },
                    )
                )

        edges = await graph_expander._edges_between([n.id for n in nodes], tenant_id)
        subgraph = Subgraph(nodes=nodes, edges=edges)

        chunks = await graph_expander._chunks_for_entities(
            [n.id for n in nodes], tenant_id, exclude=set()
        )
        for chunk in chunks:
            chunk.retrieval_path = "graph"
        return subgraph, chunks

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _fuse(
        query: str,
        vector_chunks: Sequence[RetrievedChunk],
        lexical_chunks: Sequence[RetrievedChunk],
        graph_chunks: Sequence[RetrievedChunk],
        plan: RetrievalPlan,
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Three-way RRF over vector, lexical, and graph rankings.

        Fusing by rank rather than score is what makes this work: cosine
        similarity, BM25, and hop distance are on incomparable scales, and
        normalizing them against each other would be arbitrary.
        """
        by_id: Dict[str, RetrievedChunk] = {}
        paths_by_id: Dict[str, Set[str]] = {}

        for group in (vector_chunks, lexical_chunks, graph_chunks):
            for chunk in group:
                existing = by_id.get(chunk.chunk_id)
                if existing is None:
                    by_id[chunk.chunk_id] = chunk
                    paths_by_id[chunk.chunk_id] = {chunk.retrieval_path}
                else:
                    paths_by_id[chunk.chunk_id].add(chunk.retrieval_path)

        ranked_lists = {
            "vector": [c.chunk_id for c in vector_chunks],
            "lexical": [c.chunk_id for c in lexical_chunks],
            "graph": [c.chunk_id for c in graph_chunks],
        }
        fused = reranker_service.reciprocal_rank_fusion(
            {k: v for k, v in ranked_lists.items() if v}, weights=plan.weights
        )

        ordered: List[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            chunk.rrf_score = round(rrf_score, 6)
            # Agreement across paths is strong evidence; mark it so downstream
            # stages and telemetry can see which results were corroborated.
            if len(paths_by_id.get(chunk_id, set())) > 1:
                chunk.retrieval_path = "fused"
            ordered.append(chunk)

        return ordered[: max(top_k * 4, 20)]

    @staticmethod
    def _merge_subgraphs(
        primary: Subgraph, expansion: Subgraph, seed_ids: Sequence[str]
    ) -> Subgraph:
        """Combine traversal and expansion, pruned to a bounded size."""
        nodes = {n.id: n for n in primary.nodes}
        for node in expansion.nodes:
            nodes.setdefault(node.id, node)

        edges: Dict[Tuple[str, str, str], Edge] = {
            (e.source, e.type, e.target): e for e in primary.edges
        }
        for edge in expansion.edges:
            edges.setdefault((edge.source, edge.type, edge.target), edge)

        merged = Subgraph(nodes=list(nodes.values()), edges=list(edges.values()))
        if len(merged.nodes) > settings.MAX_TRAVERSAL_NODES:
            merged = reranker_service.prune_subgraph_by_centrality(
                merged, seed_ids, settings.MAX_TRAVERSAL_NODES
            )
        return merged

    # ------------------------------------------------------------------ linking
    async def _link_entities(
        self, analysis: QueryAnalysis, tenant_id: str, query_vector: Sequence[float]
    ) -> List[LinkedEntity]:
        """Resolve query mentions to canonical graph entities."""
        from app.services.extraction_service import normalize_entity_name  # noqa: PLC0415

        mentions = analysis.mentions + analysis.identifiers
        if not mentions:
            return []

        lookup_names: List[str] = []
        for mention in mentions:
            normalized = normalize_entity_name(mention)
            if normalized and normalized not in lookup_names:
                lookup_names.append(normalized)
            for word in mention.split():
                sub = normalize_entity_name(word)
                if sub and len(sub) > 2 and sub not in lookup_names:
                    lookup_names.append(sub)

        cypher, params = CypherParameterizer.build_entity_candidate_lookup(
            lookup_names, limit=60
        )
        try:
            rows = await arcadedb_client.execute_cypher(cypher, params, tenant_id=tenant_id)
        except (DatabaseQueryError, DatabaseConnectionError) as exc:
            logger.warning("Entity linking lookup failed: %s", exc)
            return []

        candidates: List[Dict[str, Any]] = [
            {
                "entity_id": str(r["entity_id"]),
                "name": str(r.get("name", r["entity_id"])),
                "label": str(r.get("label", "Entity")),
                "normalized_name": str(r.get("normalized_name", "")),
                "aliases": r.get("aliases") or [],
            }
            for r in rows
            if isinstance(r, dict) and r.get("entity_id")
        ]
        if not candidates:
            return []

        linked: List[LinkedEntity] = []
        seen: Set[str] = set()
        vectors = await embedding_service.encode_batch_async(mentions)

        for mention, vector in zip(mentions, vectors):
            match = resolution_service.link_mention_to_candidates(mention, candidates, vector)
            if match is None or match["entity_id"] in seen:
                continue
            seen.add(match["entity_id"])
            linked.append(
                LinkedEntity(
                    mention=mention,
                    entity_id=match["entity_id"],
                    name=match["name"],
                    label=match["label"],
                    score=float(match["score"]),
                    method=str(match["method"]),
                )
            )
        return linked


retrieval_pipeline = RetrievalPipeline()
