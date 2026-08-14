"""Retrieval quality metrics.

Standard IR measures, implemented directly so the numbers are auditable rather
than opaque library output.

  Recall@k  - of the relevant items, how many appeared in the top k.
              The primary metric: an item not retrieved cannot be recovered by
              any downstream LLM.
  MRR       - reciprocal rank of the first relevant hit. Rewards ranking the
              right answer first, not merely including it.
  nDCG@k    - discounted cumulative gain, normalized. Credits every relevant item
              with a positional discount, so ordering across the whole list counts.
  P@k       - precision: what fraction of what we returned was actually relevant.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence


# ------------------------------------------------------------------ core metrics
def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant items present in the top k."""
    if not relevant:
        return 1.0  # nothing to find; vacuously satisfied
    top_k = set(retrieved[:k])
    hits = sum(1 for item in set(relevant) if item in top_k)
    return hits / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if not retrieved:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    return sum(1 for item in top_k if item in relevant_set) / len(top_k)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant item; 0 if none present."""
    relevant_set = set(relevant)
    for idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / idx
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    relevant_set = set(relevant)
    total = 0.0
    for idx, item in enumerate(retrieved[:k], start=1):
        if item in relevant_set:
            total += 1.0 / math.log2(idx + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """DCG normalized by the best achievable ordering."""
    if not relevant:
        return 1.0
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(set(relevant)), k) + 1))
    if ideal == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def hit_rate(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1.0 if at least one relevant item was retrieved."""
    if not relevant:
        return 1.0
    return 1.0 if set(retrieved) & set(relevant) else 0.0


# ------------------------------------------------------------------ aggregation
@dataclass
class QuestionResult:
    """Scored outcome for one question."""

    question: str
    category: str
    requires_multi_hop: bool
    hops: int

    retrieved_entities: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit: float = 0.0

    text_match: bool = False
    entities_linked: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    vector_hits: int = 0
    fallback_used: bool = False

    latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)

    # Set when the vector-only ablation is run, to isolate the graph's contribution.
    vector_only_hit: Optional[float] = None

    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """A question passes when it found what it was supposed to find."""
        if self.error:
            return False
        if self.category == "negative":
            # Negative questions must NOT surface entities from this tenant.
            return not self.retrieved_entities or self.fallback_used
        return self.hit > 0 or self.text_match


@dataclass
class AggregateMetrics:
    """Summary across a set of question results."""

    count: int = 0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate: float = 0.0
    pass_rate: float = 0.0

    entity_linking_rate: float = 0.0
    fallback_rate: float = 0.0
    error_rate: float = 0.0

    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": self.count,
            "recall_at_5": round(self.recall_at_5, 4),
            "recall_at_10": round(self.recall_at_10, 4),
            "precision_at_5": round(self.precision_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_5": round(self.ndcg_at_5, 4),
            "hit_rate": round(self.hit_rate, 4),
            "pass_rate": round(self.pass_rate, 4),
            "entity_linking_rate": round(self.entity_linking_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 2),
            "latency_p95_ms": round(self.latency_p95_ms, 2),
            "latency_p99_ms": round(self.latency_p99_ms, 2),
            "latency_mean_ms": round(self.latency_mean_ms, 2),
        }


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Explicit so small-N behaviour is predictable."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(pct / 100.0 * len(ordered))) - 1))
    return ordered[idx]


def aggregate(results: Sequence[QuestionResult]) -> AggregateMetrics:
    """Summarize a set of question results."""
    if not results:
        return AggregateMetrics()

    scored = [r for r in results if r.error is None]
    n = len(results)
    n_scored = max(1, len(scored))
    latencies = [r.latency_ms for r in scored if r.latency_ms > 0]

    return AggregateMetrics(
        count=n,
        recall_at_5=sum(r.recall_at_5 for r in scored) / n_scored,
        recall_at_10=sum(r.recall_at_10 for r in scored) / n_scored,
        precision_at_5=sum(r.precision_at_5 for r in scored) / n_scored,
        mrr=sum(r.mrr for r in scored) / n_scored,
        ndcg_at_5=sum(r.ndcg_at_5 for r in scored) / n_scored,
        hit_rate=sum(r.hit for r in scored) / n_scored,
        pass_rate=sum(1 for r in results if r.passed) / n,
        entity_linking_rate=sum(1 for r in scored if r.entities_linked > 0) / n_scored,
        fallback_rate=sum(1 for r in scored if r.fallback_used) / n_scored,
        error_rate=sum(1 for r in results if r.error) / n,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        latency_mean_ms=statistics.fmean(latencies) if latencies else 0.0,
    )


def score_question(
    result: QuestionResult, retrieved: Sequence[str], expected: Sequence[str]
) -> QuestionResult:
    """Populate all IR metrics on a result in place."""
    result.retrieved_entities = list(retrieved)
    result.expected_entities = list(expected)
    result.recall_at_5 = recall_at_k(retrieved, expected, 5)
    result.recall_at_10 = recall_at_k(retrieved, expected, 10)
    result.precision_at_5 = precision_at_k(retrieved, expected, 5)
    result.mrr = reciprocal_rank(retrieved, expected)
    result.ndcg_at_5 = ndcg_at_k(retrieved, expected, 5)
    result.hit = hit_rate(retrieved, expected)
    return result


def multi_hop_advantage(results: Sequence[QuestionResult]) -> Dict[str, float]:
    """Quantify what the graph path adds over vector-only retrieval.

    This is the number that justifies the architecture: if graph and vector-only
    score the same on multi-hop questions, the graph is not earning its cost.
    """
    multi_hop = [r for r in results if r.requires_multi_hop and r.error is None]
    if not multi_hop:
        return {"multi_hop_count": 0}

    with_ablation = [r for r in multi_hop if r.vector_only_hit is not None]
    hybrid = sum(r.hit for r in multi_hop) / len(multi_hop)

    out: Dict[str, float] = {
        "multi_hop_count": len(multi_hop),
        "hybrid_hit_rate": round(hybrid, 4),
        "graph_nodes_mean": round(
            sum(r.graph_nodes for r in multi_hop) / len(multi_hop), 2
        ),
    }

    if with_ablation:
        vector_only = sum(r.vector_only_hit or 0.0 for r in with_ablation) / len(with_ablation)
        hybrid_on_same = sum(r.hit for r in with_ablation) / len(with_ablation)
        out.update(
            {
                "vector_only_hit_rate": round(vector_only, 4),
                "hybrid_hit_rate_same_subset": round(hybrid_on_same, 4),
                "graph_lift": round(hybrid_on_same - vector_only, 4),
                "questions_only_graph_solved": sum(
                    1 for r in with_ablation if r.hit > 0 and (r.vector_only_hit or 0) == 0
                ),
            }
        )
    return out


def by_category(results: Sequence[QuestionResult]) -> Dict[str, AggregateMetrics]:
    """Break results down by question category."""
    buckets: Dict[str, List[QuestionResult]] = {}
    for r in results:
        buckets.setdefault(r.category, []).append(r)
    return {cat: aggregate(rs) for cat, rs in sorted(buckets.items())}
