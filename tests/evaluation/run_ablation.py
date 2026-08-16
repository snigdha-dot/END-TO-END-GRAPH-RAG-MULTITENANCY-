"""Runs the retrieval ablation study and produces the comparison report.

    python -m tests.evaluation.run_ablation
    python -m tests.evaluation.run_ablation --configs A,H
    python -m tests.evaluation.run_ablation --limit 20

Covers tasks 6, 7, 8 and 12: the eight-configuration ablation, per-category graph
lift, the slowest-query profile, and the final comparison table.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.lexical_search import lexical_search_service
from app.services.reranker_service import reranker_service
from tests.evaluation.ablation import (
    ABLATION_CONFIGS,
    AblationRunner,
    QueryMetrics,
    aggregate_config,
    graph_lift,
)
from tests.evaluation.edge_case_suite import ALL_QUERIES, PassRule


def slowest_queries(
    results: Sequence[QueryMetrics], limit: int = 10
) -> List[Dict[str, Any]]:
    """The slowest queries with their per-stage breakdown.

    Aggregate percentiles say latency is bad; this says which queries and which
    stage, which is the difference between knowing and guessing.
    """
    ordered = sorted(
        [r for r in results if r.error is None],
        key=lambda r: r.total_latency_ms,
        reverse=True,
    )
    return [
        {
            "rank": index,
            "query": r.query[:70],
            "category": r.category,
            "tenant": r.tenant,
            "config": r.config,
            "vector_candidates": r.vector_candidates,
            "vector_ms": r.stage_latencies.get("vector_ms", 0.0),
            "lexical_ms": r.stage_latencies.get("lexical_ms", 0.0),
            "graph_ms": r.stage_latencies.get("graph_ms", 0.0),
            "fusion_ms": r.stage_latencies.get("fusion_ms", 0.0),
            "rerank_ms": r.stage_latencies.get("rerank_ms", 0.0),
            "total_ms": round(r.total_latency_ms, 1),
        }
        for index, r in enumerate(ordered[:limit], start=1)
    ]


def render_report(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    summary = payload["configurations"]
    environment = payload["environment"]

    lines.append("# Retrieval Ablation Study")
    lines.append("")
    lines.append(f"**Generated:** {payload['generated_at']}")
    lines.append(
        f"**Queries:** {payload['query_count']} "
        f"({payload['scored_query_count']} with IR ground truth)"
    )
    lines.append("")

    lines.append("## Configuration comparison")
    lines.append("")
    lines.append(
        "| Configuration | R@1 | R@5 | R@10 | P@5 | MRR | NDCG@10 | P50 | P95 | P99 |"
    )
    lines.append(
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for config in ABLATION_CONFIGS:
        data = summary.get(config.key)
        if not data:
            continue
        lines.append(
            f"| {config.name} | {data['recall_at_1']:.3f} | {data['recall_at_5']:.3f} | "
            f"{data['recall_at_10']:.3f} | {data['precision_at_5']:.3f} | "
            f"{data['mrr']:.3f} | {data['ndcg_at_10']:.3f} | "
            f"{data['p50_ms']:.0f}ms | {data['p95_ms']:.0f}ms | {data['p99_ms']:.0f}ms |"
        )
    lines.append("")

    lift = payload.get("graph_lift", {})
    if lift:
        lines.append("## Graph lift")
        lines.append("")
        lines.append(
            "Hybrid Recall@10 minus vector-only Recall@10. The number that decides "
            "whether the graph earns its cost, reported per category because the "
            "graph should help relationship and multi-hop questions specifically."
        )
        lines.append("")
        lines.append(
            "| Category | N | Vector R@10 | Hybrid R@10 | Lift | Solved only by graph |"
        )
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for category, data in lift.items():
            if not data.get("queries"):
                continue
            lines.append(
                f"| {category} | {data['queries']} | "
                f"{data['vector_recall_at_10']:.3f} | {data['hybrid_recall_at_10']:.3f} | "
                f"**{data['graph_lift']:+.3f}** | "
                f"{data.get('solved_only_by_graph', '—')} |"
            )
        lines.append("")

    slow = payload.get("slowest_queries", [])
    if slow:
        lines.append("## Ten slowest queries")
        lines.append("")
        lines.append(
            "| # | Query | Category | Cand. | Vector | BM25 | Graph | Rerank | Total |"
        )
        lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in slow:
            query = row["query"][:44].replace("|", "\\|")
            lines.append(
                f"| {row['rank']} | `{query}` | {row['category']} | "
                f"{row['vector_candidates']} | {row['vector_ms']:.0f}ms | "
                f"{row['lexical_ms']:.0f}ms | {row['graph_ms']:.0f}ms | "
                f"{row['rerank_ms']:.0f}ms | **{row['total_ms']:.0f}ms** |"
            )
        lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| :--- | :--- |")
    for key, value in environment.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Reading the ablation")
    lines.append("")
    lines.append(
        "- **F vs G** isolates RRF: identical paths, fusion the only difference."
    )
    lines.append(
        "- **G vs H** isolates the cross-encoder: identical retrieval and fusion, "
        "reranking the only difference."
    )
    lines.append(
        "- **A vs E** isolates the graph on otherwise identical vector retrieval."
    )
    lines.append(
        "- Quality and latency are reported together on purpose: optimising either "
        "alone produces a system that is fast and wrong, or accurate and unusable."
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval ablation study")
    parser.add_argument("--configs", help="Comma-separated config keys, e.g. A,E,H")
    parser.add_argument("--limit", type=int, help="Cap queries per configuration")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument(
        "--dataset",
        choices=["ayurveda", "tmdb"],
        default="ayurveda",
        help="Which query set to run. 'tmdb' is the relational corpus where the "
             "graph has cross-document structure to traverse.",
    )
    args = parser.parse_args()

    configs = list(ABLATION_CONFIGS)
    if args.configs:
        wanted = {k.strip().upper() for k in args.configs.split(",")}
        configs = [c for c in configs if c.key in wanted]

    # Only queries with IR ground truth take part: an abstention case has no
    # relevant chunk, so including it would drag every configuration's recall
    # toward the same meaningless number.
    if args.dataset == "tmdb":
        from tests.evaluation.tmdb_dataset import TMDB_QUERIES  # noqa: PLC0415

        source = TMDB_QUERIES
    else:
        source = ALL_QUERIES

    cases = [q for q in source if q.has_ground_truth and q.rule is not PassRule.CLARIFIES]
    if args.limit:
        cases = cases[: args.limit]

    await arcadedb_client.start()
    if not await arcadedb_client.is_ready():
        print("ArcadeDB is not reachable.")
        await arcadedb_client.close()
        return 2

    environment = {
        "semantic_embeddings": embedding_service.is_semantic,
        "embedding_model": embedding_service.model_label,
        "embedding_version": embedding_service.embedding_version,
        "cross_encoder_active": reranker_service.has_cross_encoder,
        "reranker_model": reranker_service.model_label,
    }

    print("=" * 78)
    print(f"ABLATION STUDY — {len(configs)} configurations × {len(cases)} queries")
    for key, value in environment.items():
        print(f"  {key:24s}: {value}")
    print("=" * 78)

    runner = AblationRunner(top_k=10)
    all_results: Dict[str, List[QueryMetrics]] = {}
    summary: Dict[str, Any] = {}

    for config in configs:
        print(f"\n  [{config.key}] {config.name}")
        started = time.perf_counter()
        results = []
        for case in cases:
            results.append(await runner.score_query(case, config))
        all_results[config.key] = results
        summary[config.key] = aggregate_config(results)
        data = summary[config.key]
        print(
            f"      R@5={data['recall_at_5']:.3f} R@10={data['recall_at_10']:.3f} "
            f"MRR={data['mrr']:.3f} NDCG@10={data['ndcg_at_10']:.3f} "
            f"p95={data['p95_ms']:.0f}ms  ({time.perf_counter() - started:.0f}s)"
        )

    lift = {}
    if "A" in all_results and "H" in all_results:
        lift = graph_lift(all_results["A"], all_results["H"])

    combined = [r for results in all_results.values() for r in results]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "query_count": len(cases),
        "scored_query_count": sum(
            1 for r in all_results.get(configs[0].key, []) if r.scored
        ),
        "configurations": summary,
        "graph_lift": lift,
        "slowest_queries": slowest_queries(combined, limit=10),
        "per_query": {
            key: [
                {
                    "query": r.query,
                    "category": r.category,
                    "tenant": r.tenant,
                    "route": r.route,
                    "recall_at_1": r.recall_at_1,
                    "recall_at_5": r.recall_at_5,
                    "recall_at_10": r.recall_at_10,
                    "precision_at_5": r.precision_at_5,
                    "mrr": r.mrr,
                    "ndcg_at_10": r.ndcg_at_10,
                    "retrieved_chunk_ids": r.retrieved_chunk_ids,
                    "expected_chunk_ids": r.expected_chunk_ids,
                    "retrieved_entities": r.retrieved_entities[:20],
                    "expected_entities": r.expected_entities,
                    "retrieved_relationships": r.retrieved_relationships[:20],
                    "expected_relationships": r.expected_relationships,
                    "retrieval_latency_ms": round(r.retrieval_latency_ms, 1),
                    "total_latency_ms": round(r.total_latency_ms, 1),
                    "stage_latencies": r.stage_latencies,
                    "error": r.error,
                }
                for r in results
            ]
            for key, results in all_results.items()
        },
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out_dir / f"ablation_{stamp}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = render_report(payload)
    (out_dir / f"ablation_{stamp}.md").write_text(report, encoding="utf-8")
    (out_dir / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")

    print()
    print("=" * 78)
    for config in configs:
        data = summary[config.key]
        print(
            f"  {config.key} {config.name:<16} R@10={data['recall_at_10']:.3f} "
            f"MRR={data['mrr']:.3f} NDCG={data['ndcg_at_10']:.3f} "
            f"p95={data['p95_ms']:>6.0f}ms"
        )
    if lift.get("overall"):
        print(f"\n  GRAPH LIFT (overall): {lift['overall']['graph_lift']:+.3f}")
    print("=" * 78)
    print(f"Report: {out_dir / 'ABLATION_REPORT.md'}")

    await arcadedb_client.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
