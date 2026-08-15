"""Re-index tenants with the current embedding model.

Necessary whenever EMBEDDING_VERSION changes: vectors written by a previous model
are rejected by the read path, so they must be rewritten rather than left in
place. Chunk ids are stable, so MERGE updates each chunk's vector without
duplicating it or disturbing the entity graph built from it.

    python -m tests.evaluation.reindex_tenants
    python -m tests.evaluation.reindex_tenants --tenant ayurveda_v2
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any, Dict, List

from app.core.config import settings
from app.core.tenant_context import TenantContext, tenant_scope
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service
from app.services.lexical_search import lexical_search_service

BATCH = 64


async def list_tenants() -> List[str]:
    """Discover provisioned tenants from the server's database list."""
    client = await arcadedb_client._get_client()  # noqa: SLF001 - admin utility
    response = await client.get(f"{arcadedb_client.base_url}/api/v1/databases")
    names = response.json().get("result", [])
    return [
        n.removeprefix("tenant_").removesuffix("_kb")
        for n in names
        if n.startswith("tenant_") and n.endswith("_kb")
    ]


async def reindex_tenant(tenant_id: str) -> Dict[str, Any]:
    """Recompute and rewrite every chunk vector for one tenant."""
    started = time.perf_counter()
    ctx = TenantContext(tenant_id=tenant_id, api_key_id="reindex", request_id="reindex")

    with tenant_scope(ctx):
        rows = await arcadedb_client.execute_sql(
            "SELECT chunk_id, text, chunk_kind, embedding_version FROM Chunk LIMIT :limit",
            {"limit": 100_000},
            tenant_id=tenant_id,
        )

    chunks = [
        r for r in rows
        if isinstance(r, dict) and r.get("chunk_id") and str(r.get("text", "")).strip()
    ]
    if not chunks:
        return {"tenant": tenant_id, "chunks": 0, "updated": 0, "seconds": 0.0}

    stale = sum(
        1 for c in chunks
        if c.get("embedding_version") != embedding_service.embedding_version
    )

    updated = 0
    for start in range(0, len(chunks), BATCH):
        window = chunks[start : start + BATCH]
        vectors = embedding_service.encode_batch([str(c["text"]) for c in window])

        statements = [
            {
                "command": (
                    "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                    "SET c.embedding = $embedding, "
                    "c.embedding_version = $version, c.embedding_dim = $dim"
                ),
                "params": {
                    "chunk_id": str(chunk["chunk_id"]),
                    "embedding": list(vector),
                    "version": embedding_service.embedding_version,
                    "dim": len(vector),
                },
            }
            for chunk, vector in zip(window, vectors)
        ]
        with tenant_scope(ctx):
            updated += await arcadedb_client.execute_batch(statements, tenant_id=tenant_id)

        print(f"    {min(start + BATCH, len(chunks))}/{len(chunks)} chunks re-embedded")

    # The BM25 index caches chunk text; a re-index invalidates that assumption.
    lexical_search_service.invalidate(tenant_id)

    return {
        "tenant": tenant_id,
        "chunks": len(chunks),
        "stale_before": stale,
        "updated": updated,
        "seconds": round(time.perf_counter() - started, 1),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Re-index tenant embeddings")
    parser.add_argument("--tenant", help="Re-index only this tenant")
    args = parser.parse_args()

    await arcadedb_client.start()
    if not await arcadedb_client.is_ready():
        print("ArcadeDB is not reachable.")
        await arcadedb_client.close()
        return 2

    print("=" * 70)
    print("RE-INDEX")
    print(f"  model            : {settings.EMBEDDING_MODEL}")
    print(f"  embedding version: {embedding_service.embedding_version}")
    print(f"  semantic         : {embedding_service.is_semantic}")
    print("=" * 70)

    if not embedding_service.is_semantic:
        print()
        print("Refusing to re-index with the lexical fallback active: it would")
        print("stamp hashing vectors with a version implying semantic quality.")
        await arcadedb_client.close()
        return 2

    tenants = [args.tenant] if args.tenant else await list_tenants()
    results = []
    for tenant in tenants:
        print(f"\n  {tenant}")
        result = await reindex_tenant(tenant)
        results.append(result)
        print(
            f"    done: {result['updated']} statements, "
            f"{result.get('stale_before', 0)} were stale, {result['seconds']}s"
        )

    print()
    print("=" * 70)
    for result in results:
        print(
            f"  {result['tenant']:<18} {result['chunks']:>5} chunks  "
            f"{result['seconds']:>6.1f}s"
        )
    print("=" * 70)

    await arcadedb_client.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
