"""End-to-end evaluation harness.

Provisions tenants, ingests labelled corpora, runs the question sets, executes the
isolation battery, and writes a full report.

    python -m tests.evaluation.run_evaluation
    python -m tests.evaluation.run_evaluation --skip-ingest      # reuse existing data
    python -m tests.evaluation.run_evaluation --no-ablation      # faster
    python -m tests.evaluation.run_evaluation --out reports/

Requires a running ArcadeDB. It refuses to emit numbers otherwise, rather than
reporting figures produced by a fallback path — a report that cannot distinguish a
working system from a broken one is worse than no report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings
from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.extraction_service import extraction_service
from app.services.graph_schema_service import graph_schema_service
from app.services.ingestion_service import ingestion_service
from app.services.reranker_service import reranker_service
from app.services.retrieval_service import retrieval_service
from tests.evaluation.dataset import ALL_FIXTURES, TenantFixture
from tests.evaluation.isolation_suite import IsolationSuite
from tests.evaluation.metrics import (
    QuestionResult,
    aggregate,
    by_category,
    multi_hop_advantage,
    score_question,
)
from tests.evaluation.report import render_report


def _ctx(tenant_id: str, request_id: str = "eval") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        api_key_id=f"eval_{tenant_id}",
        request_id=request_id,
        auth_method="evaluation_harness",
    )


async def preflight() -> Dict[str, Any]:
    """Verify the environment can produce meaningful numbers."""
    await arcadedb_client.start()
    ready = await arcadedb_client.is_ready()
    return {
        "arcadedb_ready": ready,
        "arcadedb_url": settings.ARCADEDB_URL,
        "semantic_embeddings": embedding_service.is_semantic,
        "embedding_model": embedding_service.model_label,
        "cross_encoder_active": reranker_service.has_cross_encoder,
        "reranker_model": reranker_service.model_label,
        "ner_backend": extraction_service.active_backend,
        "extractor_model": extraction_service.model_label,
    }


async def provision_and_ingest(fixture: TenantFixture) -> Dict[str, Any]:
    """Provision the tenant schema and ingest its labelled corpus."""
    started = time.perf_counter()
    await graph_schema_service.provision_tenant(fixture.tenant_id)

    totals = {"documents": 0, "chunks": 0, "entities": 0, "relationships": 0}
    ctx = _ctx(fixture.tenant_id, "ingest")
    with tenant_scope(ctx):
        for doc_id, content in fixture.documents.items():
            result = await ingestion_service.ingest_document(
                ctx=ctx, doc_id=doc_id, content=content, metadata={"source": "evaluation"}
            )
            totals["documents"] += 1
            totals["chunks"] += result.get("chunks_created", 0)
            totals["entities"] += result.get("entities_extracted", 0)
            totals["relationships"] += result.get("relationships_created", 0)

    totals["ingest_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return totals


async def run_question_set(
    fixture: TenantFixture, ablation: bool = True
) -> List[QuestionResult]:
    """Run every labelled question for one tenant and score the outcomes."""
    results: List[QuestionResult] = []

    for question in fixture.questions:
        result = QuestionResult(
            question=question.question,
            category=question.category,
            requires_multi_hop=question.requires_multi_hop,
            hops=question.hops,
        )
        ctx = _ctx(fixture.tenant_id, f"q_{len(results)}")

        started = time.perf_counter()
        try:
            with tenant_scope(ctx):
                response = await retrieval_service.execute_retrieval(
                    ctx=ctx, query=question.question, max_depth=3, top_k=10
                )
            result.latency_ms = (time.perf_counter() - started) * 1000

            subgraph = response["subgraph"]
            passages = response["passages"]
            diagnostics = response["telemetry"].get("retrieval_diagnostics", {})

            retrieved_ids = [n.id for n in subgraph.nodes]
            score_question(result, retrieved_ids, question.expected_entities)

            joined = " ".join(passages).lower()
            result.text_match = bool(question.expected_text) and all(
                fragment.lower() in joined for fragment in question.expected_text
            )

            result.entities_linked = diagnostics.get("seed_entity_count", 0)
            result.graph_nodes = diagnostics.get("graph_nodes", 0)
            result.graph_edges = diagnostics.get("graph_edges", 0)
            result.vector_hits = diagnostics.get("vector_hits", 0)
            result.fallback_used = diagnostics.get("fallback_used", False)
            result.stage_latencies = response["telemetry"].get("latency_breakdown_ms", {})

            # Ablation: re-run with the graph path disabled to isolate its value.
            if ablation and question.requires_multi_hop:
                with tenant_scope(ctx):
                    vector_only = await retrieval_service.execute_retrieval(
                        ctx=ctx,
                        query=question.question,
                        max_depth=1,
                        top_k=10,
                        include_vector_search=True,
                        disable_graph_path=True,
                    )
                vo_text = " ".join(vector_only["passages"]).lower()
                vo_hit = 1.0 if (
                    question.expected_text
                    and all(f.lower() in vo_text for f in question.expected_text)
                ) else 0.0
                result.vector_only_hit = vo_hit

        except Exception as exc:  # noqa: BLE001 - a failed question is a datum
            result.error = f"{type(exc).__name__}: {exc}"
            result.latency_ms = (time.perf_counter() - started) * 1000

        results.append(result)

    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description="Graph RAG evaluation harness")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse existing data")
    parser.add_argument("--no-ablation", action="store_true", help="Skip vector-only comparison")
    parser.add_argument("--no-isolation", action="store_true", help="Skip the isolation battery")
    parser.add_argument("--out", default="reports", help="Output directory")
    args = parser.parse_args()

    print("=" * 78)
    print("GRAPH RAG MULTI-TENANCY EVALUATION")
    print("=" * 78)

    env = await preflight()
    for key, value in env.items():
        print(f"  {key:24s}: {value}")
    print()

    if not env["arcadedb_ready"]:
        print("ArcadeDB is NOT reachable at", settings.ARCADEDB_URL)
        print()
        print("Refusing to produce metrics without a live database: numbers from the")
        print("fallback path would describe the harness, not the system.")
        print()
        print("  1. wsl --install --no-distribution   (Administrator, then reboot)")
        print("  2. docker compose up -d arcadedb")
        print("  3. re-run this harness")
        await arcadedb_client.close()
        return 2

    if not env["semantic_embeddings"]:
        print("WARNING: running in lexical-fallback mode (no ML dependencies).")
        print("         Quality metrics will understate real performance.")
        print("         Install with: pip install -r requirements-ml.txt")
        print()

    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "tenants": {},
        "ingestion": {},
    }

    # ---------------------------------------------------------------- ingest
    if not args.skip_ingest:
        for fixture in ALL_FIXTURES:
            print(f"Ingesting corpus for '{fixture.tenant_id}'...")
            report["ingestion"][fixture.tenant_id] = await provision_and_ingest(fixture)
            stats = report["ingestion"][fixture.tenant_id]
            print(
                f"  {stats['documents']} docs, {stats['chunks']} chunks, "
                f"{stats['entities']} entities, {stats['relationships']} relations "
                f"({stats['ingest_ms']:.0f}ms)"
            )
        print()

    # ---------------------------------------------------------------- quality
    all_results: List[QuestionResult] = []
    for fixture in ALL_FIXTURES:
        print(f"Evaluating '{fixture.tenant_id}' ({len(fixture.questions)} questions)...")
        results = await run_question_set(fixture, ablation=not args.no_ablation)
        all_results.extend(results)

        agg = aggregate(results)
        report["tenants"][fixture.tenant_id] = {
            "aggregate": agg.to_dict(),
            "by_category": {k: v.to_dict() for k, v in by_category(results).items()},
            "multi_hop": multi_hop_advantage(results),
            "questions": [
                {
                    "question": r.question,
                    "category": r.category,
                    "passed": r.passed,
                    "hit": r.hit,
                    "recall_at_5": round(r.recall_at_5, 4),
                    "mrr": round(r.mrr, 4),
                    "ndcg_at_5": round(r.ndcg_at_5, 4),
                    "entities_linked": r.entities_linked,
                    "graph_nodes": r.graph_nodes,
                    "fallback_used": r.fallback_used,
                    "latency_ms": round(r.latency_ms, 2),
                    "vector_only_hit": r.vector_only_hit,
                    "error": r.error,
                }
                for r in results
            ],
        }
        print(
            f"  pass_rate={agg.pass_rate:.2%}  recall@5={agg.recall_at_5:.3f}  "
            f"MRR={agg.mrr:.3f}  p95={agg.latency_p95_ms:.0f}ms"
        )
    print()

    report["overall"] = aggregate(all_results).to_dict()
    report["overall_multi_hop"] = multi_hop_advantage(all_results)

    # ---------------------------------------------------------------- isolation
    if not args.no_isolation:
        print("Running multi-tenancy isolation battery...")
        suite = IsolationSuite(ALL_FIXTURES)
        await suite.run_all(include_live=True)
        report["isolation"] = suite.summary()
        verdict = report["isolation"]["isolation_verdict"]
        print(
            f"  {report['isolation']['passed']}/{report['isolation']['total_checks']} "
            f"checks passed  ->  {verdict}"
        )
        print()

    # ---------------------------------------------------------------- output
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"evaluation_{stamp}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = out_dir / f"evaluation_{stamp}.md"
    md_path.write_text(render_report(report), encoding="utf-8")

    latest = out_dir / "EVALUATION_REPORT.md"
    latest.write_text(render_report(report), encoding="utf-8")

    print(f"JSON report : {json_path}")
    print(f"Markdown    : {md_path}")
    print(f"Latest      : {latest}")

    await arcadedb_client.close()

    isolation_failed = (
        not args.no_isolation and report["isolation"]["isolation_verdict"] == "FAIL"
    )
    return 1 if isolation_failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
