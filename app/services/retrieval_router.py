"""Retrieval router: decides which paths a query is worth spending.

Every path costs something different — a graph traversal is orders of magnitude
more expensive than a BM25 scan — so running all of them on every query wastes
the budget on paths that cannot contribute.

    LOCAL    vector + lexical + graph. An entity anchor exists, so traversal has
             somewhere to start.
    GLOBAL   community search, plus vector as a safety net. No anchor means
             traversal has nothing to seed from, so it is skipped rather than
             run and discarded.
    LEXICAL  lexical first, vector as backup. Exact identifiers and quoted
             phrases are what dense vectors are worst at.
    HYBRID   everything. Chosen when the signals disagree, so the fallback is
             breadth rather than a guess.

The plan is data, not control flow: it says what to run and how much of each,
and the pipeline executes it. That keeps routing decisions inspectable in
telemetry instead of buried in branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.core.config import settings
from app.services.query_understanding import QueryAnalysis, QueryIntent


@dataclass
class RetrievalPlan:
    """Which paths to run, and with what budget."""

    intent: QueryIntent
    use_vector: bool = True
    use_lexical: bool = True
    use_graph: bool = True
    use_community: bool = False
    use_graph_expansion: bool = True

    vector_k: int = 20
    lexical_k: int = 20
    graph_depth: int = 2
    expansion_hops: int = 1
    community_k: int = 5

    # Fusion weights per path, applied to the RRF contribution. A path the query
    # analysis favours should influence the ranking more than one kept as backup.
    weights: Dict[str, float] = field(default_factory=dict)
    rationale: List[str] = field(default_factory=list)

    @property
    def active_paths(self) -> List[str]:
        paths = []
        if self.use_vector:
            paths.append("vector")
        if self.use_lexical:
            paths.append("lexical")
        if self.use_graph:
            paths.append("graph")
        if self.use_community:
            paths.append("community")
        return paths

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "active_paths": self.active_paths,
            "vector_k": self.vector_k,
            "lexical_k": self.lexical_k,
            "graph_depth": self.graph_depth,
            "expansion_hops": self.expansion_hops,
            "graph_expansion": self.use_graph_expansion,
            "weights": {k: round(v, 2) for k, v in self.weights.items()},
            "rationale": self.rationale,
        }


class RetrievalRouter:
    """Turns a query analysis into an execution plan."""

    def plan(self, analysis: QueryAnalysis, top_k: int = 5) -> RetrievalPlan:
        candidate_k = max(top_k * 4, 20)

        if analysis.intent is QueryIntent.GLOBAL:
            return self._global_plan(analysis, candidate_k, top_k)
        if analysis.intent is QueryIntent.LEXICAL:
            return self._lexical_plan(analysis, candidate_k)
        if analysis.intent is QueryIntent.LOCAL:
            return self._local_plan(analysis, candidate_k)
        return self._hybrid_plan(analysis, candidate_k, top_k)

    # ------------------------------------------------------------------ plans
    def _local_plan(self, analysis: QueryAnalysis, candidate_k: int) -> RetrievalPlan:
        return RetrievalPlan(
            intent=analysis.intent,
            use_vector=True,
            use_lexical=True,
            use_graph=analysis.has_anchor,
            use_community=False,
            use_graph_expansion=True,
            vector_k=candidate_k,
            lexical_k=candidate_k,
            graph_depth=min(analysis.suggested_hops + 1, settings.MAX_TRAVERSAL_DEPTH),
            # Expansion starts from entities in the *retrieved* chunks, not from
            # the query's seeds, so it reaches material the initial traversal
            # could not. That is worth one hop even on single-hop questions.
            expansion_hops=2 if analysis.suggested_hops > 1 else 1,
            weights={"graph": 1.4, "vector": 1.0, "lexical": 0.8},
            rationale=[
                "entity anchor present; graph traversal can seed"
                if analysis.has_anchor
                else "no anchor resolved; graph path skipped",
                f"multi-hop phrasing → depth {analysis.suggested_hops + 1}"
                if analysis.suggested_hops > 1
                else "single-hop phrasing",
            ],
        )

    def _global_plan(
        self, analysis: QueryAnalysis, candidate_k: int, top_k: int
    ) -> RetrievalPlan:
        return RetrievalPlan(
            intent=analysis.intent,
            use_vector=True,
            use_lexical=False,
            # No entity anchor, so a traversal would have nothing to start from.
            use_graph=False,
            use_community=True,
            use_graph_expansion=False,
            vector_k=candidate_k,
            community_k=max(top_k, 5),
            weights={"community": 1.5, "vector": 1.0},
            rationale=[
                "thematic query; community reports summarise regions of the graph",
                "graph traversal skipped: no entity to seed from",
            ],
        )

    def _lexical_plan(self, analysis: QueryAnalysis, candidate_k: int) -> RetrievalPlan:
        return RetrievalPlan(
            intent=analysis.intent,
            use_vector=True,
            use_lexical=True,
            use_graph=analysis.has_anchor,
            use_community=False,
            use_graph_expansion=False,
            vector_k=max(candidate_k // 2, 10),
            lexical_k=candidate_k,
            graph_depth=1,
            expansion_hops=0,
            weights={"lexical": 1.6, "vector": 0.7, "graph": 0.6},
            rationale=[
                "exact identifiers or quoted phrases; dense vectors match these poorly",
                "lexical weighted above vector",
            ],
        )

    def _hybrid_plan(
        self, analysis: QueryAnalysis, candidate_k: int, top_k: int
    ) -> RetrievalPlan:
        return RetrievalPlan(
            intent=analysis.intent,
            use_vector=True,
            use_lexical=True,
            use_graph=analysis.has_anchor,
            use_community=not analysis.has_anchor,
            use_graph_expansion=analysis.has_anchor,
            vector_k=candidate_k,
            lexical_k=candidate_k,
            graph_depth=min(analysis.suggested_hops + 1, settings.MAX_TRAVERSAL_DEPTH),
            expansion_hops=1,
            community_k=max(top_k, 3),
            weights={"vector": 1.0, "lexical": 1.0, "graph": 1.0, "community": 0.8},
            rationale=[
                "signals disagree; running all viable paths rather than guessing",
            ],
        )


retrieval_router = RetrievalRouter()
