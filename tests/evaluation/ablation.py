"""Retrieval ablation study: which components actually earn their cost.

Runs the same 100 queries through eight configurations, from single-path to full
hybrid, and measures quality and latency together. Optimising for either alone is
how systems end up fast and wrong, or accurate and unusable.

    A  Vector only
    B  BM25 only
    C  Graph only
    D  Vector + BM25
    E  Vector + Graph
    F  Vector + BM25 + Graph          (concatenated, no fusion)
    G  Vector + BM25 + Graph + RRF    (rank fusion)
    H  Full hybrid + RRF + cross-encoder reranker

F and G differ only in fusion, which isolates what RRF contributes; G and H
differ only in reranking, which isolates the cross-encoder. Without those two
pairs the study would show that "more is better" without saying which part.

The router is bypassed here on purpose: it would choose paths per query, and the
study needs each configuration applied uniformly to be comparable.
"""
from __future__ import annotations

import asyncio
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.exceptions import DatabaseConnectionError, DatabaseQueryError
from app.core.security import CypherParameterizer
from app.core.tenant_context import TenantContext, tenant_scope
from app.models.graph import RetrievedChunk, Subgraph
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.lexical_search import lexical_search_service
from app.services.query_understanding import query_understanding
from app.services.reranker_service import reranker_service
from app.services.retrieval_pipeline import retrieval_pipeline
from tests.evaluation.edge_case_suite import Category, EdgeCaseQuery


@dataclass(frozen=True)
class AblationConfig:
    """One point in the ablation grid."""

    key: str
    name: str
    use_vector: bool = False
    use_lexical: bool = False
    use_graph: bool = False
    use_rrf: bool = False
    use_reranker: bool = False


ABLATION_CONFIGS: List[AblationConfig] = [
    AblationConfig("A", "Vector", use_vector=True),
    AblationConfig("B", "BM25", use_lexical=True),
    AblationConfig("C", "Graph", use_graph=True),
    AblationConfig("D", "Vector+BM25", use_vector=True, use_lexical=True),
    AblationConfig("E", "Vector+Graph", use_vector=True, use_graph=True),
    AblationConfig("F", "Hybrid", use_vector=True, use_lexical=True, use_graph=True),
    AblationConfig("G", "Hybrid+RRF", use_vector=True, use_lexical=True,
                   use_graph=True, use_rrf=True),
    AblationConfig("H", "Full+Reranker", use_vector=True, use_lexical=True,
                   use_graph=True, use_rrf=True, use_reranker=True),
]


# ------------------------------------------------------------------- IR metrics
def recall_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0  # nothing to find; vacuously satisfied
    hits = sum(1 for item in set(retrieved[:k]) if item in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 1.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


@dataclass
class QueryMetrics:
    """Full metric set for one query under one configuration."""

    query: str
    category: str
    tenant: str
    config: str
    route: str = ""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    retrieved_chunk_ids: List[str] = field(default_factory=list)
    expected_chunk_ids: List[str] = field(default_factory=list)
    retrieved_entities: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)
    retrieved_relationships: List[str] = field(default_factory=list)
    expected_relationships: List[str] = field(default_factory=list)

    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    vector_candidates: int = 0
    scored: bool = False
    error: Optional[str] = None


class AblationRunner:
    """Executes each configuration over the query set and scores the results."""

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._chunk_cache: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------ ground truth
    async def resolve_ground_truth(
        self, case: EdgeCaseQuery
    ) -> Tuple[Set[str], List[str]]:
        """Map content markers to the chunk ids that currently contain them.

        Chunk ids are positional and change whenever chunking configuration does,
        so ground truth is expressed as content and resolved against the live
        index at run time. That keeps the same ground truth valid across the
        chunking A/B.
        """
        if not case.relevant_chunk_markers:
            return set(), []

        rows = self._chunk_cache.get(case.tenant)
        if rows is None:
            with tenant_scope(
                TenantContext(tenant_id=case.tenant, api_key_id="ablation",
                              request_id="ablation")
            ):
                rows = await arcadedb_client.execute_sql(
                    "SELECT chunk_id, text FROM Chunk "
                    "WHERE chunk_kind != 'community_report' LIMIT :limit",
                    {"limit": 100_000},
                    tenant_id=case.tenant,
                )
            rows = [r for r in rows if isinstance(r, dict) and r.get("chunk_id")]
            self._chunk_cache[case.tenant] = rows

        markers = [m.lower() for m in case.relevant_chunk_markers]
        matched = {
            str(row["chunk_id"])
            for row in rows
            if any(marker in str(row.get("text", "")).lower() for marker in markers)
        }
        return matched, sorted(matched)

    def invalidate_ground_truth(self) -> None:
        """Drop cached chunks after a re-ingest changes them."""
        self._chunk_cache.clear()

    # ---------------------------------------------------------------- retrieval
    async def _run_config(
        self, case: EdgeCaseQuery, config: AblationConfig
    ) -> Tuple[List[RetrievedChunk], Subgraph, Dict[str, float], int]:
        """Run one configuration, bypassing the router for comparability."""
        ctx = TenantContext(
            tenant_id=case.tenant, api_key_id="ablation", request_id="ablation"
        )
        stages: Dict[str, float] = {}
        query_vector = embedding_service.encode_query(case.query)

        vector_chunks: List[RetrievedChunk] = []
        lexical_chunks: List[RetrievedChunk] = []
        graph_chunks: List[RetrievedChunk] = []
        subgraph = Subgraph()
        candidate_k = max(self.top_k * 3, 30)

        with tenant_scope(ctx):
            if config.use_vector:
                started = time.perf_counter()
                vector_chunks = await retrieval_pipeline._vector_search(  # noqa: SLF001
                    query_vector, case.tenant, candidate_k
                )
                stages["vector_ms"] = (time.perf_counter() - started) * 1000

            if config.use_lexical:
                started = time.perf_counter()
                lexical_chunks = await lexical_search_service.search(
                    case.query, case.tenant, candidate_k
                )
                stages["lexical_ms"] = (time.perf_counter() - started) * 1000

            if config.use_graph:
                started = time.perf_counter()
                analysis = query_understanding.analyze(case.query)
                linked = await retrieval_pipeline._link_entities(  # noqa: SLF001
                    analysis, case.tenant, query_vector
                )
                seed_ids = [e.entity_id for e in linked]
                if seed_ids:
                    subgraph, graph_chunks = await retrieval_pipeline._graph_search(  # noqa: SLF001
                        seed_ids, case.tenant, ctx, ctx.schema, 2
                    )
                stages["graph_ms"] = (time.perf_counter() - started) * 1000

        # ------------------------------------------------------------ combine
        if config.use_rrf:
            started = time.perf_counter()
            ranked_lists = {
                name: [c.chunk_id for c in group]
                for name, group in (
                    ("vector", vector_chunks),
                    ("lexical", lexical_chunks),
                    ("graph", graph_chunks),
                )
                if group
            }
            by_id = {
                c.chunk_id: c
                for group in (vector_chunks, lexical_chunks, graph_chunks)
                for c in group
            }
            fused = reranker_service.reciprocal_rank_fusion(ranked_lists)
            combined = [by_id[cid] for cid, _ in fused if cid in by_id]
            stages["fusion_ms"] = (time.perf_counter() - started) * 1000
        else:
            # Without fusion, concatenate and deduplicate by first appearance.
            seen: Set[str] = set()
            combined = []
            for group in (vector_chunks, lexical_chunks, graph_chunks):
                for chunk in group:
                    if chunk.chunk_id not in seen:
                        seen.add(chunk.chunk_id)
                        combined.append(chunk)

        if config.use_reranker and combined:
            started = time.perf_counter()
            combined = reranker_service.rerank_chunks(case.query, combined, self.top_k)
            stages["rerank_ms"] = (time.perf_counter() - started) * 1000

        return combined[: self.top_k], subgraph, stages, len(vector_chunks)

    # ------------------------------------------------------------------ scoring
    async def score_query(
        self, case: EdgeCaseQuery, config: AblationConfig
    ) -> QueryMetrics:
        metrics = QueryMetrics(
            query=case.query,
            category=case.category.value,
            tenant=case.tenant,
            config=config.key,
            route=config.name,
        )

        started = time.perf_counter()
        try:
            chunks, subgraph, stages, candidates = await self._run_config(case, config)
        except Exception as exc:  # noqa: BLE001 - a failed config is a datum
            metrics.error = f"{type(exc).__name__}: {exc}"
            metrics.total_latency_ms = (time.perf_counter() - started) * 1000
            return metrics

        metrics.total_latency_ms = (time.perf_counter() - started) * 1000
        metrics.retrieval_latency_ms = sum(
            v for k, v in stages.items() if k.endswith("_ms")
        )
        metrics.stage_latencies = {k: round(v, 2) for k, v in stages.items()}
        metrics.vector_candidates = candidates

        metrics.retrieved_chunk_ids = [c.chunk_id for c in chunks]
        metrics.retrieved_entities = [n.id for n in subgraph.nodes]
        metrics.retrieved_relationships = [
            f"{e.source}-{e.type}->{e.target}" for e in subgraph.edges
        ]

        relevant, expected_ids = await self.resolve_ground_truth(case)
        metrics.expected_chunk_ids = expected_ids
        metrics.expected_entities = list(case.relevant_entities)
        metrics.expected_relationships = list(case.relevant_relationships)

        # Only queries with ground truth contribute to IR aggregates. Scoring an
        # abstention case as recall 0 would penalise correct behaviour.
        if relevant:
            metrics.scored = True
            retrieved = metrics.retrieved_chunk_ids
            metrics.recall_at_1 = recall_at_k(retrieved, relevant, 1)
            metrics.recall_at_5 = recall_at_k(retrieved, relevant, 5)
            metrics.recall_at_10 = recall_at_k(retrieved, relevant, 10)
            metrics.precision_at_5 = precision_at_k(retrieved, relevant, 5)
            metrics.mrr = reciprocal_rank(retrieved, relevant)
            metrics.ndcg_at_10 = ndcg_at_k(retrieved, relevant, 10)

        return metrics


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(pct / 100 * len(ordered))) - 1))
    return ordered[index]


def aggregate_config(results: Sequence[QueryMetrics]) -> Dict[str, Any]:
    """Summarize one configuration across all queries."""
    scored = [r for r in results if r.scored and r.error is None]
    latencies = [r.total_latency_ms for r in results if r.error is None]

    def mean(attr: str) -> float:
        values = [getattr(r, attr) for r in scored]
        return round(statistics.fmean(values), 4) if values else 0.0

    return {
        "queries": len(results),
        "scored_queries": len(scored),
        "errors": sum(1 for r in results if r.error),
        "recall_at_1": mean("recall_at_1"),
        "recall_at_5": mean("recall_at_5"),
        "recall_at_10": mean("recall_at_10"),
        "precision_at_5": mean("precision_at_5"),
        "mrr": mean("mrr"),
        "ndcg_at_10": mean("ndcg_at_10"),
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "mean_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
    }


def graph_lift(
    vector_results: Sequence[QueryMetrics], hybrid_results: Sequence[QueryMetrics]
) -> Dict[str, Any]:
    """Graph lift = hybrid Recall@10 - vector-only Recall@10, per category.

    The number that decides whether the graph earns its cost. Reported per
    category because the graph should help relationship and multi-hop questions
    specifically; a uniform lift across all categories would suggest the
    comparison is measuring something else.
    """
    graph_categories = [
        Category.RELATIONSHIP.value,
        Category.MULTI_HOP.value,
        Category.COMPARISON.value,
        Category.GLOBAL.value,
    ]

    by_query_vector = {r.query: r for r in vector_results if r.scored}
    by_query_hybrid = {r.query: r for r in hybrid_results if r.scored}

    output: Dict[str, Any] = {}
    for category in graph_categories:
        vector_scores = [
            r.recall_at_10 for q, r in by_query_vector.items()
            if r.category == category
        ]
        hybrid_scores = [
            r.recall_at_10 for q, r in by_query_hybrid.items()
            if r.category == category
        ]
        if not vector_scores or not hybrid_scores:
            output[category] = {"queries": 0, "note": "no scored queries in category"}
            continue

        vector_mean = statistics.fmean(vector_scores)
        hybrid_mean = statistics.fmean(hybrid_scores)
        solved_only_by_graph = sum(
            1 for q, h in by_query_hybrid.items()
            if h.category == category
            and h.recall_at_10 > 0
            and by_query_vector.get(q, h).recall_at_10 == 0
        )
        output[category] = {
            "queries": len(hybrid_scores),
            "vector_recall_at_10": round(vector_mean, 4),
            "hybrid_recall_at_10": round(hybrid_mean, 4),
            "graph_lift": round(hybrid_mean - vector_mean, 4),
            "solved_only_by_graph": solved_only_by_graph,
        }

    all_vector = [r.recall_at_10 for r in vector_results if r.scored]
    all_hybrid = [r.recall_at_10 for r in hybrid_results if r.scored]
    if all_vector and all_hybrid:
        output["overall"] = {
            "queries": len(all_hybrid),
            "vector_recall_at_10": round(statistics.fmean(all_vector), 4),
            "hybrid_recall_at_10": round(statistics.fmean(all_hybrid), 4),
            "graph_lift": round(
                statistics.fmean(all_hybrid) - statistics.fmean(all_vector), 4
            ),
        }
    return output
