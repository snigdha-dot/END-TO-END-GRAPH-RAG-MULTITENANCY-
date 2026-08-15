"""Side-by-side latency (ms) and price ($) benchmark harness (plan section 7, row 4).

Runs without ArcadeDB by stubbing the database layer, so the telemetry contract is
verifiable in CI. The live-data equivalent lives in tests/integration.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from app.core.telemetry import CONTRACT_STEPS
from app.core.tenant_context import TenantContext, tenant_scope
from app.services import retrieval_service as retrieval_module
from app.services.retrieval_service import retrieval_service

BENCHMARK_QUERIES = [
    ("movies_bot", "Which other films did the director of Inception make?"),
    ("movies_bot", "Who starred in Interstellar?"),
    ("ai_trends_bot", "What models build on the Transformer architecture?"),
    ("ai_trends_bot", "Which organization released GPT-4?"),
]


class _StubDB:
    """Minimal in-memory stand-in for ArcadeDB with a small movies graph."""

    ENTITIES = [
        {"entity_id": "canon_film_inception", "name": "Inception", "label": "Film",
         "normalized_name": "inception", "aliases": []},
        {"entity_id": "canon_person_christopher_nolan", "name": "Christopher Nolan",
         "label": "Person", "normalized_name": "christopher_nolan", "aliases": []},
        {"entity_id": "canon_film_interstellar", "name": "Interstellar", "label": "Film",
         "normalized_name": "interstellar", "aliases": []},
    ]

    async def execute_cypher(self, cypher: str, params=None, *, tenant_id=None, language="cypher"):
        params = params or {}
        if "normalized_name IN $names" in cypher:
            names = set(params.get("names", []))
            return [e for e in self.ENTITIES if e["normalized_name"] in names]
        if "start.entity_id IN $start_nodes" in cypher:
            # ArcadeDB has no path functions, so traversal projects endpoint pairs
            # and a second query recovers the typed edges between them.
            return [
                {
                    "source_id": "canon_film_inception", "source_name": "Inception",
                    "source_label": "Film",
                    "target_id": "canon_person_christopher_nolan",
                    "target_name": "Christopher Nolan", "target_label": "Person",
                },
                {
                    "source_id": "canon_film_inception", "source_name": "Inception",
                    "source_label": "Film",
                    "target_id": "canon_film_interstellar",
                    "target_name": "Interstellar", "target_label": "Film",
                },
            ]
        if "type(r) AS rel_type" in cypher:
            return [
                {
                    "source_id": "canon_person_christopher_nolan",
                    "rel_type": "DIRECTED",
                    "target_id": "canon_film_inception",
                    "confidence": 0.92,
                },
                {
                    "source_id": "canon_person_christopher_nolan",
                    "rel_type": "DIRECTED",
                    "target_id": "canon_film_interstellar",
                    "confidence": 0.92,
                },
            ]
        if "MENTIONED_IN" in cypher:
            return [
                {"chunk_id": "c1", "text": "Christopher Nolan directed Inception and Interstellar.",
                 "parent_doc_id": "d1", "section_path": ["Inception"]}
            ]
        return []

    async def execute_sql(self, sql: str, params=None, *, tenant_id=None):
        if "FROM Chunk" in sql:
            return [
                {"chunk_id": "c1", "text": "Christopher Nolan directed Inception.",
                 "parent_doc_id": "d1", "section_path": [], "embedding": [0.1] * 384}
            ]
        return []


@pytest.fixture
def stub_db(monkeypatch):
    stub = _StubDB()
    monkeypatch.setattr(retrieval_module, "arcadedb_client", stub)
    return stub


def _print_report(tenant: str, query: str, result: Dict[str, Any]) -> None:
    telemetry = result["telemetry"]
    print("\n" + "=" * 78)
    print("TEAM B SIDE-BY-SIDE RETRIEVAL TELEMETRY")
    print("=" * 78)
    print(f"Tenant DB      : tenant_{tenant}_kb")
    print(f"Query          : {query}")
    print(f"Nodes / Edges  : {len(result['subgraph'].nodes)} / {len(result['subgraph'].edges)}")
    print(f"Passages       : {len(result['passages'])}")

    diagnostics = telemetry.get("retrieval_diagnostics", {})
    print(f"Seeds linked   : {diagnostics.get('seed_entity_count', 0)}")
    print(f"Fallback used  : {diagnostics.get('fallback_used')}")

    print("-" * 78)
    print(f"{'LATENCY BREAKDOWN':<44}{'ms':>12}")
    print("-" * 78)
    for step, value in telemetry["latency_breakdown_ms"].items():
        print(f"  {step:<42}{value:>12.2f}")

    print("-" * 78)
    print(f"{'MODEL':<28}{'STEP':<24}{'TOKENS':>10}{'COST USD':>14}")
    print("-" * 78)
    for call in telemetry["model_cost_breakdown"]["models_called"]:
        print(
            f"  {call['model_name']:<26}{call['step']:<24}"
            f"{call['tokens_used']['total_tokens']:>10}{call['cost_usd']:>14.8f}"
        )
    cost = telemetry["model_cost_breakdown"]
    print("-" * 78)
    print(f"{'TOTAL TOKENS':<52}{cost['total_tokens']:>10}")
    print(f"{'TOTAL REQUEST COST (USD)':<52}{cost['total_request_cost_usd']:>10.8f}")
    print("=" * 78)


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant,query", BENCHMARK_QUERIES)
async def test_side_by_side_telemetry_report(stub_db, tenant, query):
    ctx = TenantContext(tenant_id=tenant, api_key_id="bench", request_id="bench")
    with tenant_scope(ctx):
        result = await retrieval_service.execute_retrieval(ctx=ctx, query=query, top_k=5)

    _print_report(tenant, query, result)

    telemetry = result["telemetry"]
    for step in CONTRACT_STEPS:
        assert step in telemetry["latency_breakdown_ms"]
    assert telemetry["latency_breakdown_ms"]["total_retrieval_latency"] > 0
    assert "models_called" in telemetry["model_cost_breakdown"]
    assert telemetry["model_cost_breakdown"]["total_request_cost_usd"] >= 0


@pytest.mark.asyncio
async def test_multi_hop_query_links_real_seeds(stub_db):
    """Regression guard: the linker must find "Inception", not stopwords."""
    ctx = TenantContext(tenant_id="movies_bot", api_key_id="bench", request_id="bench")
    with tenant_scope(ctx):
        result = await retrieval_service.execute_retrieval(
            ctx=ctx, query="Which other films did the director of Inception make?", top_k=5
        )

    diagnostics = result["telemetry"]["retrieval_diagnostics"]
    assert diagnostics["seed_entity_count"] > 0
    assert not diagnostics["fallback_used"]

    linked = [e["name"].lower() for e in diagnostics["linked_entities"]]
    assert any("inception" in name for name in linked)


@pytest.mark.asyncio
async def test_graph_passages_verbalize_relationships(stub_db):
    """Multi-hop answers live in edges, not in any single chunk of text."""
    ctx = TenantContext(tenant_id="movies_bot", api_key_id="bench", request_id="bench")
    with tenant_scope(ctx):
        result = await retrieval_service.execute_retrieval(
            ctx=ctx, query="Which other films did the director of Inception make?", top_k=5
        )

    joined = " ".join(result["passages"]).lower()
    assert "nolan" in joined
    assert "interstellar" in joined


@pytest.mark.asyncio
async def test_all_foss_models_report_zero_cost(stub_db):
    ctx = TenantContext(tenant_id="movies_bot", api_key_id="bench", request_id="bench")
    with tenant_scope(ctx):
        result = await retrieval_service.execute_retrieval(ctx=ctx, query="Inception", top_k=3)
    assert result["telemetry"]["model_cost_breakdown"]["total_request_cost_usd"] == 0.0
