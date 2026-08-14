"""Renders evaluation results as a markdown report with explicit pass/fail gates.

Gates are stated as thresholds rather than prose so a run either clears the bar or
does not. Isolation is the only hard gate: retrieval quality is a tuning problem,
cross-tenant leakage is a correctness failure.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Production gates. Isolation is absolute; quality gates are targets.
GATES: Dict[str, Dict[str, Any]] = {
    "isolation_verdict": {"expect": "PASS", "hard": True,
                          "label": "Zero cross-tenant leakage"},
    "injection_defence": {"expect": 1.0, "hard": True,
                          "label": "All injection payloads rejected"},
    "recall_at_5": {"expect": 0.70, "hard": False, "label": "Recall@5 >= 0.70"},
    "mrr": {"expect": 0.60, "hard": False, "label": "MRR >= 0.60"},
    "entity_linking_rate": {"expect": 0.70, "hard": False,
                            "label": "Entity linking rate >= 0.70"},
    "fallback_rate": {"expect": 0.30, "hard": False, "label": "Fallback rate <= 0.30",
                      "invert": True},
    "error_rate": {"expect": 0.0, "hard": True, "label": "Zero unhandled errors"},
    "latency_p95_ms": {"expect": 500.0, "hard": False, "label": "p95 latency <= 500ms",
                       "invert": True},
}


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _evaluate_gates(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check each gate against the report, returning rows for the summary table."""
    overall = report.get("overall", {})
    isolation = report.get("isolation", {})
    rows: List[Dict[str, Any]] = []

    for key, gate in GATES.items():
        if key == "isolation_verdict":
            actual: Any = isolation.get("isolation_verdict", "NOT RUN")
            passed = actual == gate["expect"]
        elif key == "injection_defence":
            cats = isolation.get("by_category", {})
            inj = cats.get("injection_defence", {})
            total = inj.get("passed", 0) + inj.get("failed", 0)
            actual = round(inj.get("passed", 0) / total, 4) if total else "NOT RUN"
            passed = actual == 1.0 if total else False
        else:
            actual = overall.get(key)
            if actual is None:
                actual = "NOT RUN"
                passed = False
            elif gate.get("invert"):
                passed = actual <= gate["expect"]
            else:
                passed = actual >= gate["expect"]

        rows.append(
            {
                "label": gate["label"],
                "expected": gate["expect"],
                "actual": actual,
                "passed": passed,
                "hard": gate["hard"],
            }
        )
    return rows


def render_report(report: Dict[str, Any]) -> str:
    """Render the full markdown report."""
    lines: List[str] = []
    env = report.get("environment", {})
    overall = report.get("overall", {})
    isolation = report.get("isolation", {})

    lines.append("# Graph RAG Multi-Tenancy Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {report.get('generated_at', 'unknown')}")
    lines.append("")

    # ------------------------------------------------------------- verdict
    gate_rows = _evaluate_gates(report)
    hard_failures = [r for r in gate_rows if r["hard"] and not r["passed"]]
    soft_failures = [r for r in gate_rows if not r["hard"] and not r["passed"]]

    if hard_failures:
        headline = "FAIL — one or more correctness gates did not hold"
    elif soft_failures:
        headline = "CONDITIONAL PASS — correctness gates held; quality targets missed"
    else:
        headline = "PASS — all gates cleared"

    lines.append(f"## Verdict: {headline}")
    lines.append("")

    if not env.get("semantic_embeddings", False):
        lines.append(
            "> **Degraded mode.** ML dependencies are not installed, so embeddings, "
            "reranking, and NER ran on lexical fallbacks. Quality metrics below "
            "understate real performance; isolation and security results are unaffected."
        )
        lines.append("")

    # ------------------------------------------------------------- gates
    lines.append("## Production gates")
    lines.append("")
    lines.append("| Gate | Type | Target | Actual | Result |")
    lines.append("| :--- | :--- | ---: | ---: | :--- |")
    for row in gate_rows:
        kind = "hard" if row["hard"] else "target"
        lines.append(
            f"| {row['label']} | {kind} | {row['expected']} | {row['actual']} | "
            f"{_verdict(row['passed'])} |"
        )
    lines.append("")

    # ------------------------------------------------------------- isolation
    if isolation:
        lines.append("## Multi-tenancy isolation")
        lines.append("")
        lines.append(f"**Verdict: {isolation.get('isolation_verdict', 'NOT RUN')}** — "
                     f"{isolation.get('passed', 0)}/{isolation.get('total_checks', 0)} "
                     f"checks passed.")
        lines.append("")
        lines.append("| Category | Passed | Failed |")
        lines.append("| :--- | ---: | ---: |")
        for cat, counts in sorted(isolation.get("by_category", {}).items()):
            lines.append(f"| {cat} | {counts.get('passed', 0)} | {counts.get('failed', 0)} |")
        lines.append("")

        failures = isolation.get("failures", [])
        if failures:
            lines.append("### Failed checks")
            lines.append("")
            for failure in failures:
                lines.append(f"- **{failure['name']}** ({failure['category']}): {failure['detail']}")
            lines.append("")
        else:
            lines.append("No isolation failures. Every cross-tenant probe returned zero "
                         "foreign entities, and all injection payloads were rejected.")
            lines.append("")

    # ------------------------------------------------------------- quality
    lines.append("## Retrieval quality — overall")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| :--- | ---: |")
    for key in (
        "count", "pass_rate", "hit_rate", "recall_at_5", "recall_at_10",
        "precision_at_5", "mrr", "ndcg_at_5", "entity_linking_rate",
        "fallback_rate", "error_rate",
    ):
        if key in overall:
            lines.append(f"| {key} | {overall[key]} |")
    lines.append("")

    lines.append("## Latency")
    lines.append("")
    lines.append("| Percentile | Milliseconds |")
    lines.append("| :--- | ---: |")
    for key, label in (
        ("latency_p50_ms", "p50"), ("latency_p95_ms", "p95"),
        ("latency_p99_ms", "p99"), ("latency_mean_ms", "mean"),
    ):
        if key in overall:
            lines.append(f"| {label} | {overall[key]} |")
    lines.append("")

    # ------------------------------------------------------------- multi-hop
    mh = report.get("overall_multi_hop", {})
    if mh.get("multi_hop_count"):
        lines.append("## Multi-hop advantage")
        lines.append("")
        lines.append(
            "The question this answers: does graph traversal retrieve answers that "
            "vector-only search cannot? If the lift is zero, the graph is not earning "
            "its complexity."
        )
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| :--- | ---: |")
        for key, value in mh.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    # ------------------------------------------------------------- per tenant
    for tenant_id, data in report.get("tenants", {}).items():
        lines.append(f"## Tenant: `{tenant_id}`")
        lines.append("")
        agg = data.get("aggregate", {})
        lines.append(
            f"pass_rate **{agg.get('pass_rate')}** · recall@5 **{agg.get('recall_at_5')}** · "
            f"MRR **{agg.get('mrr')}** · nDCG@5 **{agg.get('ndcg_at_5')}** · "
            f"p95 **{agg.get('latency_p95_ms')}ms**"
        )
        lines.append("")

        cats = data.get("by_category", {})
        if cats:
            lines.append("| Category | N | Pass rate | Recall@5 | MRR |")
            lines.append("| :--- | ---: | ---: | ---: | ---: |")
            for cat, metrics in cats.items():
                lines.append(
                    f"| {cat} | {metrics.get('count')} | {metrics.get('pass_rate')} | "
                    f"{metrics.get('recall_at_5')} | {metrics.get('mrr')} |"
                )
            lines.append("")

        lines.append("<details><summary>Per-question detail</summary>")
        lines.append("")
        lines.append("| Question | Cat | Pass | R@5 | MRR | Linked | Nodes | FB | ms |")
        lines.append("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | ---: |")
        for q in data.get("questions", []):
            question = q["question"][:52].replace("|", "\\|")
            lines.append(
                f"| {question} | {q['category'][:9]} | {_verdict(q['passed'])} | "
                f"{q['recall_at_5']} | {q['mrr']} | {q['entities_linked']} | "
                f"{q['graph_nodes']} | {'Y' if q['fallback_used'] else 'N'} | "
                f"{q['latency_ms']:.0f} |"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

        failed = [q for q in data.get("questions", []) if not q["passed"]]
        if failed:
            lines.append("### Failed questions")
            lines.append("")
            for q in failed:
                reason = q["error"] or (
                    "no entities linked" if q["entities_linked"] == 0
                    else "fallback used" if q["fallback_used"]
                    else "expected entities not retrieved"
                )
                lines.append(f"- `{q['question']}` — {reason}")
            lines.append("")

    # ------------------------------------------------------------- ingestion
    ingestion = report.get("ingestion", {})
    if ingestion:
        lines.append("## Ingestion")
        lines.append("")
        lines.append("| Tenant | Docs | Chunks | Entities | Relations | ms |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
        for tenant_id, stats in ingestion.items():
            lines.append(
                f"| {tenant_id} | {stats.get('documents')} | {stats.get('chunks')} | "
                f"{stats.get('entities')} | {stats.get('relationships')} | "
                f"{stats.get('ingest_ms')} |"
            )
        lines.append("")

    # ------------------------------------------------------------- environment
    lines.append("## Environment")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("| :--- | :--- |")
    for key, value in env.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # ------------------------------------------------------------- reading it
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- **Isolation is the only hard gate that matters commercially.** A single "
        "cross-tenant leak invalidates the product regardless of retrieval quality."
    )
    lines.append(
        "- **Recall@5 is the primary quality metric.** A chunk not retrieved cannot "
        "be recovered by any downstream LLM."
    )
    lines.append(
        "- **Entity linking rate near zero means the graph path is dead** and every "
        "query is silently falling through to vector search."
    )
    lines.append(
        "- **Fallback rate is the early-warning signal.** A rising trend means "
        "retrieval is degrading before users report it."
    )
    lines.append(
        "- **Graph lift is the architectural justification.** If multi-hop questions "
        "score the same with and without traversal, the graph is cost without benefit."
    )
    lines.append("")

    return "\n".join(lines)
