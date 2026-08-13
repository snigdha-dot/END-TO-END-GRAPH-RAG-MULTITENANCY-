"""End-to-End Retrieval Pipeline Benchmark with Side-by-Side Latency (ms) & Price ($) Telemetry output."""
import pytest
import asyncio
from app.services.retrieval_service import retrieval_service

@pytest.mark.asyncio
async def test_end_to_end_retrieval_side_by_side_telemetry():
    tenant_id = "tech_support_bot"
    query = "What services fail if Auth Service goes down?"

    subgraph, passages, telemetry = await retrieval_service.execute_retrieval(
        tenant_id=tenant_id,
        query=query,
        max_depth=2,
        top_k=5
    )

    print("\n" + "="*80)
    print("📊 TEAM B SIDE-BY-SIDE RETRIEVAL TELEMETRY REPORT")
    print("="*80)
    print(f"Tenant Target DB : tenant_{tenant_id}_kb")
    print(f"Search Query     : '{query}'")
    print(f"Nodes Retrieved  : {len(subgraph.nodes)}")
    print(f"Edges Retrieved  : {len(subgraph.edges)}")
    print("-" * 80)
    print("LATENCY BREAKDOWN (ms):")
    for step, latency in telemetry["latency_breakdown_ms"].items():
        print(f"  • {step:<30}: {latency:>8.2f} ms")
    print("-" * 80)
    print("MODEL COST BREAKDOWN ($ USD):")
    model_calls = telemetry["model_cost_breakdown"]["models_called"]
    for call in model_calls:
        print(f"  • [{call['step']}] Model: {call['model_name']}")
        print(f"    Tokens: Prompt={call['tokens_used']['prompt_tokens']}, Comp={call['tokens_used']['completion_tokens']}, Total={call['tokens_used']['total_tokens']}")
        print(f"    Cost  : ${call['cost_usd']:.6f} | Duration: {call['latency_ms']:.2f} ms")
    print("-" * 80)
    print(f"TOTAL REQUEST COST (USD) : ${telemetry['model_cost_breakdown']['total_request_cost_usd']:.6f}")
    print("="*80 + "\n")

    assert "latency_breakdown_ms" in telemetry
    assert "model_cost_breakdown" in telemetry
    assert telemetry["latency_breakdown_ms"]["total_retrieval_latency"] > 0
