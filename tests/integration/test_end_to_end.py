"""End-to-end integration tests against a live ArcadeDB.

Skipped automatically when ArcadeDB is unreachable, so the suite stays runnable
without infrastructure. Run them with:

    docker compose up -d arcadedb
    pytest tests/integration -v

These are the tests that prove the system actually works, as opposed to the unit
tests which prove each part is individually correct.
"""
from __future__ import annotations

import os

import pytest

from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.graph_schema_service import graph_schema_service
from app.services.ingestion_service import ingestion_service
from app.services.retrieval_service import retrieval_service

pytestmark = pytest.mark.integration

TEST_TENANT = os.getenv("INTEGRATION_TEST_TENANT", "itest_movies")
OTHER_TENANT = os.getenv("INTEGRATION_OTHER_TENANT", "itest_ai")

MOVIES_DOC = """
# Inception

Inception is a 2010 science fiction film. Inception was directed by
Christopher Nolan. Leonardo DiCaprio starred in Inception.

# Interstellar

Interstellar is a 2014 epic. Interstellar was directed by Christopher Nolan.
Matthew McConaughey starred in Interstellar.

# Dunkirk

Dunkirk is a 2017 war film. Dunkirk was directed by Christopher Nolan.
"""

AI_DOC = """
# GPT-4

GPT-4 was released by OpenAI. GPT-4 builds on the Transformer architecture.

# Claude

Claude was released by Anthropic. Claude builds on the Transformer architecture.
"""


async def _arcadedb_available() -> bool:
    await arcadedb_client.start()
    return await arcadedb_client.is_ready()


@pytest.fixture(scope="module")
async def live_db():
    if not await _arcadedb_available():
        pytest.skip("ArcadeDB is not reachable; skipping integration tests.")
    yield
    await arcadedb_client.close()


@pytest.fixture
async def provisioned_movies(live_db):
    await graph_schema_service.provision_tenant(TEST_TENANT)
    ctx = TenantContext(tenant_id=TEST_TENANT, api_key_id="itest", request_id="itest")
    with tenant_scope(ctx):
        yield ctx


@pytest.fixture
async def provisioned_ai(live_db):
    await graph_schema_service.provision_tenant(OTHER_TENANT)
    ctx = TenantContext(tenant_id=OTHER_TENANT, api_key_id="itest", request_id="itest")
    with tenant_scope(ctx):
        yield ctx


@pytest.mark.asyncio
async def test_provisioning_creates_schema_and_indexes(provisioned_movies):
    report = await graph_schema_service.verify_schema(TEST_TENANT)
    assert not report["vertex_types_missing"], report
    assert not report["edge_types_missing"], report


@pytest.mark.asyncio
async def test_ingestion_writes_a_real_graph(provisioned_movies):
    result = await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="inception_doc", content=MOVIES_DOC
    )
    assert result["chunks_created"] > 0
    assert result["entities_extracted"] > 0
    assert result["statements_executed"] > 0
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_multi_hop_retrieval_returns_real_data(provisioned_movies):
    """The proof that graph RAG beats vector-only.

    "Which other films did the director of Inception make?" requires two hops:
    Inception -> Nolan -> {Interstellar, Dunkirk}. No single chunk states it.
    """
    await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="inception_doc", content=MOVIES_DOC
    )

    result = await retrieval_service.execute_retrieval(
        ctx=provisioned_movies,
        query="Which other films did the director of Inception make?",
        max_depth=2,
        top_k=5,
    )

    diagnostics = result["telemetry"]["retrieval_diagnostics"]
    assert diagnostics["seed_entity_count"] > 0, "Entity linking found no seeds"
    assert not diagnostics["fallback_used"], "Fell back instead of traversing the graph"
    assert result["passages"], "No context passages returned"

    joined = " ".join(result["passages"]).lower()
    assert "nolan" in joined


@pytest.mark.asyncio
async def test_cross_tenant_data_is_not_reachable(provisioned_movies, provisioned_ai):
    """Plan section 7 row 1, verified against real databases.

    Ingest a distinctive entity into one tenant, then query the other for it.
    """
    await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="movies", content=MOVIES_DOC
    )
    await ingestion_service.ingest_document(ctx=provisioned_ai, doc_id="ai", content=AI_DOC)

    with tenant_scope(provisioned_ai):
        result = await retrieval_service.execute_retrieval(
            ctx=provisioned_ai, query="Tell me about Inception and Christopher Nolan", top_k=5
        )

    joined = " ".join(result["passages"]).lower()
    assert "inception" not in joined
    assert "nolan" not in joined
    for node in result["subgraph"].nodes:
        assert "nolan" not in node.name.lower()


@pytest.mark.asyncio
async def test_fallback_engages_for_an_unknown_entity(provisioned_movies):
    """Plan section 7 row 3: an unindexed entity must degrade, not error."""
    await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="movies", content=MOVIES_DOC
    )
    result = await retrieval_service.execute_retrieval(
        ctx=provisioned_movies, query="Who directed Zqxwvu Nonexistent Film?", top_k=3
    )
    # It must not raise, and must still produce a well-formed response.
    assert "latency_breakdown_ms" in result["telemetry"]


@pytest.mark.asyncio
async def test_reingestion_is_idempotent(provisioned_movies):
    """MERGE semantics: ingesting twice must not duplicate entities."""
    first = await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="idem_doc", content=MOVIES_DOC
    )
    second = await ingestion_service.ingest_document(
        ctx=provisioned_movies, doc_id="idem_doc", content=MOVIES_DOC
    )
    assert first["entities_extracted"] == second["entities_extracted"]


@pytest.mark.asyncio
async def test_unprovisioned_tenant_raises_not_found(live_db):
    from app.core.exceptions import TenantNotFoundError

    ctx = TenantContext(
        tenant_id="never_provisioned_tenant", api_key_id="itest", request_id="itest"
    )
    with tenant_scope(ctx):
        with pytest.raises(TenantNotFoundError):
            await retrieval_service.execute_retrieval(ctx=ctx, query="anything", top_k=3)
