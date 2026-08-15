"""Runs the 100-query edge case suite and reports per-category results.

    python -m tests.evaluation.run_edge_cases
    python -m tests.evaluation.run_edge_cases --category multi_hop
    python -m tests.evaluation.run_edge_cases --out reports/

Scoring is category-aware because categories fail differently. A no-answer query
passes by returning *nothing*; a relationship query passes by returning graph
edges; an isolation probe passes by finding nothing across a tenant boundary.
Scoring every category as "did we retrieve something" would mark abstention
failures as successes — which is the failure mode that matters most here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.exceptions import SecurityViolationError
from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.reranker_service import reranker_service
from app.services.retrieval_pipeline import retrieval_pipeline
from tests.evaluation.edge_case_suite import (
    ALL_QUERIES,
    CATEGORY_PURPOSE,
    Category,
    EdgeCaseQuery,
    PassRule,
)


@dataclass
class QueryOutcome:
    """What happened for one query, and whether it counts as a pass."""

    query: str
    category: str
    tenant: str
    rule: str
    passed: bool = False
    reason: str = ""

    intent: str = ""
    active_paths: List[str] = field(default_factory=list)
    vector_hits: int = 0
    lexical_hits: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    expansion_chunks: int = 0
    community_hits: int = 0
    fallback_used: bool = False
    passages: int = 0
    context_tokens: int = 0

    latency_ms: float = 0.0
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    error: Optional[str] = None


class EdgeCaseRunner:
    """Executes the suite and scores each query by its category's rule."""

    async def run_query(self, case: EdgeCaseQuery) -> QueryOutcome:
        outcome = QueryOutcome(
            query=case.query,
            category=case.category.value,
            tenant=case.tenant,
            rule=case.rule.value,
        )
        ctx = TenantContext(
            tenant_id=case.tenant, api_key_id="edge_suite", request_id="edge_suite"
        )

        started = time.perf_counter()
        try:
            with tenant_scope(ctx):
                result = await retrieval_pipeline.retrieve(
                    ctx=ctx,
                    query=case.query,
                    top_k=5,
                    # Cases that carry prior turns test resolution against context
                    # rather than clarification; without passing it the pipeline
                    # correctly asks for clarification and the case scores as a
                    # failure of the harness, not of the system.
                    conversation_context=case.conversation_context or None,
                )
            outcome.latency_ms = (time.perf_counter() - started) * 1000
        except SecurityViolationError as exc:
            outcome.latency_ms = (time.perf_counter() - started) * 1000
            # For adversarial queries this is the desired behaviour, not a failure.
            if case.rule is PassRule.REJECTS:
                outcome.passed = True
                outcome.reason = f"rejected: {exc.detail[:80]}"
            else:
                outcome.error = f"SecurityViolationError: {exc.detail[:80]}"
                outcome.reason = "unexpectedly rejected"
            return outcome
        except Exception as exc:  # noqa: BLE001 - a crash is a datum, not an abort
            outcome.latency_ms = (time.perf_counter() - started) * 1000
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.reason = "raised an unexpected exception"
            return outcome

        diagnostics = result["telemetry"].get("retrieval_diagnostics", {})
        telemetry = result["telemetry"]

        outcome.intent = diagnostics.get("query_analysis", {}).get("intent", "")
        outcome.active_paths = diagnostics.get("retrieval_plan", {}).get("active_paths", [])
        outcome.vector_hits = diagnostics.get("vector_hits", 0)
        outcome.lexical_hits = diagnostics.get("lexical_hits", 0)
        outcome.graph_nodes = diagnostics.get("graph_nodes", 0)
        outcome.graph_edges = diagnostics.get("graph_edges", 0)
        outcome.expansion_chunks = diagnostics.get("expansion_chunks", 0)
        outcome.community_hits = diagnostics.get("community_hits", 0)
        outcome.fallback_used = diagnostics.get("fallback_used", False)
        outcome.passages = len(result.get("passages", []))
        outcome.context_tokens = diagnostics.get("context", {}).get("total_tokens", 0)
        outcome.stage_latencies = telemetry.get("latency_breakdown_ms", {})
        outcome.cost_usd = telemetry.get("model_cost_breakdown", {}).get(
            "total_request_cost_usd", 0.0
        )

        passed, reason = self._score(case, result, outcome)
        outcome.passed = passed
        outcome.reason = reason
        return outcome

    def _score(
        self, case: EdgeCaseQuery, result: Dict[str, Any], outcome: QueryOutcome
    ) -> tuple[bool, str]:
        """Apply the category's pass rule."""
        joined = " ".join(result.get("passages", [])).lower()

        if case.rule is PassRule.REJECTS:
            # Reaching here means the payload was executed rather than refused.
            return False, "adversarial query was accepted, not rejected"

        if case.rule is PassRule.ABSTAINS:
            leaked = [t for t in case.forbid_text if t.lower() in joined]
            if leaked:
                return False, f"surfaced forbidden term(s): {', '.join(leaked)}"
            return True, "correctly returned no matching content"

        if case.rule is PassRule.ROUTES_TO:
            if outcome.intent == case.expect_intent:
                return True, f"routed to '{outcome.intent}' as expected"
            return False, f"routed to '{outcome.intent}', expected '{case.expect_intent}'"

        if case.rule is PassRule.RETRIEVES_GRAPH:
            if outcome.graph_edges >= max(1, case.min_graph_edges):
                return True, f"{outcome.graph_edges} graph edges returned"
            return False, (
                f"{outcome.graph_edges} graph edges, needed {max(1, case.min_graph_edges)}"
            )

        if case.rule is PassRule.RETRIEVES_TEXT:
            missing = [t for t in case.expect_text if t.lower() not in joined]
            if missing:
                return False, f"expected term(s) absent: {', '.join(missing)}"
            return True, "expected content retrieved"

        # RETRIEVES_ANY
        if outcome.passages > 0 and not outcome.fallback_used:
            return True, f"{outcome.passages} passages without fallback"
        if outcome.passages > 0:
            return True, f"{outcome.passages} passages via fallback"
        return False, "no passages returned"

    async def run(self, cases: Sequence[EdgeCaseQuery]) -> List[QueryOutcome]:
        outcomes: List[QueryOutcome] = []
        for index, case in enumerate(cases, start=1):
            outcome = await self.run_query(case)
            outcomes.append(outcome)
            mark = "PASS" if outcome.passed else "FAIL"
            print(
                f"  [{index:3d}/{len(cases)}] {mark}  {case.category.value:<22} "
                f"{outcome.latency_ms:7.0f}ms  {case.query[:44]}"
            )
        return outcomes


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    import math

    index = max(0, min(len(ordered) - 1, int(math.ceil(pct / 100 * len(ordered))) - 1))
    return ordered[index]


def aggregate(outcomes: Sequence[QueryOutcome]) -> Dict[str, Any]:
    """Summarize overall, per-category, and per-stage."""
    latencies = [o.latency_ms for o in outcomes if o.error is None]

    by_category: Dict[str, Dict[str, Any]] = {}
    for category in Category:
        group = [o for o in outcomes if o.category == category.value]
        if not group:
            continue
        group_latencies = [o.latency_ms for o in group if o.error is None]
        passed = sum(1 for o in group if o.passed)
        by_category[category.value] = {
            "purpose": CATEGORY_PURPOSE[category],
            "count": len(group),
            "passed": passed,
            "failed": len(group) - passed,
            "pass_rate": round(passed / len(group), 4),
            "latency_p50_ms": round(_percentile(group_latencies, 50), 1),
            "latency_p95_ms": round(_percentile(group_latencies, 95), 1),
            "mean_latency_ms": round(
                statistics.fmean(group_latencies) if group_latencies else 0.0, 1
            ),
        }

    stage_totals: Dict[str, List[float]] = defaultdict(list)
    for outcome in outcomes:
        for stage, value in outcome.stage_latencies.items():
            if stage != "total_retrieval_latency":
                stage_totals[stage].append(float(value))

    stages = {
        stage: {
            "mean_ms": round(statistics.fmean(values), 2),
            "p95_ms": round(_percentile(values, 95), 2),
            "share_pct": 0.0,
        }
        for stage, values in stage_totals.items()
        if values
    }
    total_stage_mean = sum(s["mean_ms"] for s in stages.values()) or 1.0
    for stage in stages.values():
        stage["share_pct"] = round(100 * stage["mean_ms"] / total_stage_mean, 1)

    passed_total = sum(1 for o in outcomes if o.passed)
    security = [
        o for o in outcomes
        if o.category in (Category.ISOLATION.value, Category.ADVERSARIAL.value)
    ]
    security_failures = [o for o in security if not o.passed]

    return {
        "total": len(outcomes),
        "passed": passed_total,
        "failed": len(outcomes) - passed_total,
        "pass_rate": round(passed_total / len(outcomes), 4) if outcomes else 0.0,
        "errors": sum(1 for o in outcomes if o.error),
        "fallback_rate": round(
            sum(1 for o in outcomes if o.fallback_used) / len(outcomes), 4
        ) if outcomes else 0.0,
        "security_checks": len(security),
        "security_failures": len(security_failures),
        "security_verdict": "PASS" if not security_failures else "FAIL",
        "latency": {
            "p50_ms": round(_percentile(latencies, 50), 1),
            "p95_ms": round(_percentile(latencies, 95), 1),
            "p99_ms": round(_percentile(latencies, 99), 1),
            "mean_ms": round(statistics.fmean(latencies) if latencies else 0.0, 1),
            "max_ms": round(max(latencies) if latencies else 0.0, 1),
        },
        "cost": {
            "total_usd": round(sum(o.cost_usd for o in outcomes), 6),
            "mean_per_query_usd": round(
                sum(o.cost_usd for o in outcomes) / len(outcomes), 8
            ) if outcomes else 0.0,
            "projected_per_1k_queries_usd": round(
                sum(o.cost_usd for o in outcomes) / len(outcomes) * 1000, 4
            ) if outcomes else 0.0,
        },
        "by_category": by_category,
        "stages": dict(sorted(stages.items(), key=lambda kv: -kv[1]["mean_ms"])),
    }


def render_report(summary: Dict[str, Any], outcomes: Sequence[QueryOutcome],
                  environment: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Edge Case Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    verdict = (
        "FAIL — security checks did not hold"
        if summary["security_verdict"] == "FAIL"
        else f"{summary['passed']}/{summary['total']} queries passed"
    )
    lines.append(f"## Result: {verdict}")
    lines.append("")

    if not environment.get("semantic_embeddings"):
        lines.append(
            "> **Degraded mode.** Embeddings are lexical hashing rather than "
            "`bge-small-en-v1.5`, and reranking is lexical overlap rather than a "
            "cross-encoder. Semantic categories understate real performance; "
            "security and routing results are unaffected."
        )
        lines.append("")

    lines.append("| Metric | Value |")
    lines.append("| :--- | ---: |")
    lines.append(f"| Total queries | {summary['total']} |")
    lines.append(f"| Passed | {summary['passed']} |")
    lines.append(f"| Failed | {summary['failed']} |")
    lines.append(f"| Pass rate | {summary['pass_rate']:.1%} |")
    lines.append(f"| Errors | {summary['errors']} |")
    lines.append(f"| Fallback rate | {summary['fallback_rate']:.1%} |")
    lines.append(f"| **Security verdict** | **{summary['security_verdict']}** |")
    lines.append("")

    lines.append("## Per-category results")
    lines.append("")
    lines.append("| # | Category | What it tests | N | Pass | Rate | p50 | p95 |")
    lines.append("| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |")
    for index, (name, data) in enumerate(summary["by_category"].items(), start=1):
        lines.append(
            f"| {index} | {name} | {data['purpose']} | {data['count']} | "
            f"{data['passed']} | {data['pass_rate']:.0%} | "
            f"{data['latency_p50_ms']:.0f}ms | {data['latency_p95_ms']:.0f}ms |"
        )
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lines.append("| Percentile | Milliseconds |")
    lines.append("| :--- | ---: |")
    for label, key in (("p50", "p50_ms"), ("p95", "p95_ms"), ("p99", "p99_ms"),
                       ("mean", "mean_ms"), ("max", "max_ms")):
        lines.append(f"| {label} | {summary['latency'][key]} |")
    lines.append("")

    lines.append("### Where the time goes")
    lines.append("")
    lines.append("| Stage | Mean | p95 | Share |")
    lines.append("| :--- | ---: | ---: | ---: |")
    for stage, data in summary["stages"].items():
        lines.append(
            f"| {stage} | {data['mean_ms']}ms | {data['p95_ms']}ms | {data['share_pct']}% |"
        )
    lines.append("")

    lines.append("## Cost")
    lines.append("")
    lines.append("| Metric | USD |")
    lines.append("| :--- | ---: |")
    lines.append(f"| Total for this run | ${summary['cost']['total_usd']:.6f} |")
    lines.append(f"| Mean per query | ${summary['cost']['mean_per_query_usd']:.8f} |")
    lines.append(
        f"| Projected per 1,000 queries | ${summary['cost']['projected_per_1k_queries_usd']:.4f} |"
    )
    lines.append("")
    if summary["cost"]["total_usd"] == 0:
        lines.append(
            "All models run locally under FOSS licences and no LLM is called, so "
            "cost is a measured zero rather than an unmeasured one. The pricing "
            "matrix is configured, so these figures become non-zero the moment a "
            "provider is enabled."
        )
        lines.append("")

    failures = [o for o in outcomes if not o.passed]
    if failures:
        lines.append("## Failures")
        lines.append("")
        lines.append("| Category | Query | Why |")
        lines.append("| :--- | :--- | :--- |")
        for outcome in failures:
            query = outcome.query[:52].replace("|", "\\|")
            lines.append(f"| {outcome.category} | `{query}` | {outcome.reason} |")
        lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| :--- | :--- |")
    for key, value in environment.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- **Security is the only hard gate.** Isolation and adversarial checks "
        "must be 100%; a single failure invalidates the deployment regardless of "
        "retrieval quality."
    )
    lines.append(
        "- **Abstention categories pass by returning nothing.** A no-answer or "
        "isolation query that surfaces content has failed, even though it "
        "retrieved successfully."
    )
    lines.append(
        "- **Graph categories are the architectural justification.** If "
        "relationship and multi-hop queries score no better than semantic ones, "
        "the graph is cost without benefit."
    )
    lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Edge case evaluation suite")
    parser.add_argument("--category", help="Run only one category")
    parser.add_argument("--out", default="reports", help="Output directory")
    parser.add_argument("--limit", type=int, help="Cap the number of queries")
    args = parser.parse_args()

    cases = list(ALL_QUERIES)
    if args.category:
        cases = [c for c in cases if c.category.value == args.category]
        if not cases:
            print(f"No queries in category '{args.category}'.")
            return 2
    if args.limit:
        cases = cases[: args.limit]

    print("=" * 78)
    print(f"EDGE CASE SUITE — {len(cases)} queries")
    print("=" * 78)

    await arcadedb_client.start()
    if not await arcadedb_client.is_ready():
        print("ArcadeDB is not reachable; refusing to report metrics without it.")
        await arcadedb_client.close()
        return 2

    environment = {
        "semantic_embeddings": embedding_service.is_semantic,
        "embedding_model": embedding_service.model_label,
        "cross_encoder_active": reranker_service.has_cross_encoder,
        "reranker_model": reranker_service.model_label,
    }
    for key, value in environment.items():
        print(f"  {key:24s}: {value}")
    print()

    runner = EdgeCaseRunner()
    outcomes = await runner.run(cases)
    summary = aggregate(outcomes)

    print()
    print("=" * 78)
    print(
        f"RESULT: {summary['passed']}/{summary['total']} passed "
        f"({summary['pass_rate']:.1%})  |  security: {summary['security_verdict']}  |  "
        f"p95 {summary['latency']['p95_ms']:.0f}ms"
    )
    print("=" * 78)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "summary": summary,
        "outcomes": [asdict(o) for o in outcomes],
    }
    (out_dir / f"edge_cases_{stamp}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    report = render_report(summary, outcomes, environment)
    (out_dir / f"edge_cases_{stamp}.md").write_text(report, encoding="utf-8")
    (out_dir / "EDGE_CASE_REPORT.md").write_text(report, encoding="utf-8")

    print(f"JSON     : {out_dir / f'edge_cases_{stamp}.json'}")
    print(f"Markdown : {out_dir / 'EDGE_CASE_REPORT.md'}")

    await arcadedb_client.close()
    return 1 if summary["security_verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
