"""In-memory vector index, per tenant.

ArcadeDB 24.11.1 does not expose HNSW index creation through SQL DDL, so vector
search has to score candidates itself. The naive form — SELECT every chunk with
its embedding, then score — measured 329ms per query against a 400-chunk tenant,
of which only 11ms was the actual scoring. The rest was transferring 400 × 384
floats over HTTP, on every single query.

So the vectors are loaded once and kept. Scoring 400 vectors in memory is a few
milliseconds; refetching them is not.

Three properties this has to preserve:

  Tenant scoping   Caches are keyed by tenant and never merged. A shared cache
                   would be a cross-tenant leak that no query-level check could
                   catch, because the data would already be in the wrong list.
  Version safety   Vectors written by a different embedding model are excluded at
                   load time, so an incompatible vector cannot be scored.
  Freshness        Ingestion invalidates the tenant's cache. A stale cache would
                   silently omit new documents, which looks like poor recall.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.models.graph import RetrievedChunk
from app.services.arcadedb_client import arcadedb_client
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)


@dataclass
class _IndexedChunk:
    """One chunk with its vector, pre-normalized for fast scoring."""

    chunk_id: str
    text: str
    parent_doc_id: str
    section_path: List[str]
    citation: str
    chunk_kind: str
    vector: List[float]
    norm: float


@dataclass
class TenantVectorIndex:
    """All searchable vectors for one tenant."""

    tenant_id: str
    chunks: List[_IndexedChunk] = field(default_factory=list)
    embedding_version: str = ""
    built_at: float = 0.0
    skipped_incompatible: int = 0

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.built_at


class VectorIndexService:
    """Loads, caches, and searches tenant vectors."""

    MAX_CHUNKS = 100_000
    PAGE_SIZE = 500

    def __init__(self) -> None:
        self._indexes: Dict[str, TenantVectorIndex] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, tenant_id: str) -> asyncio.Lock:
        if tenant_id not in self._locks:
            self._locks[tenant_id] = asyncio.Lock()
        return self._locks[tenant_id]

    # ------------------------------------------------------------------ build
    async def _load(self, tenant_id: str) -> TenantVectorIndex:
        """Page through every chunk, keeping only version-compatible vectors."""
        started = time.perf_counter()
        index = TenantVectorIndex(
            tenant_id=tenant_id,
            embedding_version=embedding_service.embedding_version,
            built_at=time.time(),
        )

        offset = 0
        while offset < self.MAX_CHUNKS:
            # ArcadeDB caps rows per response well below the requested LIMIT, so
            # paging is required to see the whole corpus rather than its head.
            rows = await arcadedb_client.execute_sql(
                "SELECT chunk_id, text, parent_doc_id, section_path, embedding, "
                "citation, chunk_kind, embedding_version FROM Chunk "
                "SKIP :offset LIMIT :limit",
                {"offset": offset, "limit": self.PAGE_SIZE},
                tenant_id=tenant_id,
                timeout_ms=settings.ARCADEDB_DDL_TIMEOUT_MS,
            )
            if not rows:
                break

            for row in rows:
                if not isinstance(row, dict):
                    continue
                vector = row.get("embedding")
                chunk_id = row.get("chunk_id")
                if not chunk_id or not isinstance(vector, list) or not vector:
                    continue
                if not embedding_service.is_compatible(row.get("embedding_version")):
                    index.skipped_incompatible += 1
                    continue

                floats = [float(x) for x in vector]
                norm = math.sqrt(sum(v * v for v in floats)) or 1.0
                index.chunks.append(
                    _IndexedChunk(
                        chunk_id=str(chunk_id),
                        text=str(row.get("text", "")),
                        parent_doc_id=str(row.get("parent_doc_id", "")),
                        section_path=list(row.get("section_path") or []),
                        citation=str(row.get("citation", "")),
                        chunk_kind=str(row.get("chunk_kind", "prose")),
                        vector=floats,
                        norm=norm,
                    )
                )

            if len(rows) < self.PAGE_SIZE:
                break
            offset += len(rows)

        logger.info(
            "Vector index built for '%s': %d chunks in %.0fms (%d skipped as "
            "incompatible)",
            tenant_id, index.size, (time.perf_counter() - started) * 1000,
            index.skipped_incompatible,
        )
        return index

    async def get_index(self, tenant_id: str, force: bool = False) -> TenantVectorIndex:
        """Return the tenant's index, building or refreshing it if needed."""
        cached = self._indexes.get(tenant_id)
        fresh = (
            cached is not None
            and not force
            and cached.embedding_version == embedding_service.embedding_version
            and cached.age_seconds < settings.VECTOR_INDEX_TTL_SECONDS
        )
        if fresh:
            return cached

        # One builder per tenant: concurrent first-requests would otherwise each
        # page the whole corpus.
        async with self._lock_for(tenant_id):
            cached = self._indexes.get(tenant_id)
            if (
                cached is not None
                and not force
                and cached.embedding_version == embedding_service.embedding_version
                and cached.age_seconds < settings.VECTOR_INDEX_TTL_SECONDS
            ):
                return cached
            index = await self._load(tenant_id)
            self._indexes[tenant_id] = index
            return index

    def invalidate(self, tenant_id: Optional[str] = None) -> None:
        """Drop cached vectors after ingestion changes the corpus."""
        if tenant_id:
            self._indexes.pop(tenant_id, None)
        else:
            self._indexes.clear()

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query_vector: Sequence[float],
        tenant_id: str,
        top_k: int = 20,
        include_community_reports: bool = False,
    ) -> List[RetrievedChunk]:
        """Cosine ranking over the tenant's cached vectors."""
        index = await self.get_index(tenant_id)
        if not index.chunks:
            return []

        query = list(query_vector)
        query_norm = math.sqrt(sum(v * v for v in query)) or 1.0

        scored: List[Tuple[float, _IndexedChunk]] = []
        for chunk in index.chunks:
            if not include_community_reports and chunk.chunk_kind == "community_report":
                continue
            if len(chunk.vector) != len(query):
                continue
            dot = sum(a * b for a, b in zip(query, chunk.vector))
            score = dot / (query_norm * chunk.norm)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: List[RetrievedChunk] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    parent_doc_id=chunk.parent_doc_id,
                    score=round(score, 4),
                    section_path=chunk.section_path,
                    retrieval_path="vector",
                    rank=rank,
                )
            )
        return results

    async def search_community_reports(
        self, query_vector: Sequence[float], tenant_id: str, top_k: int = 5
    ) -> List[RetrievedChunk]:
        """Rank only community reports, for global queries."""
        index = await self.get_index(tenant_id)
        reports = [c for c in index.chunks if c.chunk_kind == "community_report"]
        if not reports:
            return []

        query = list(query_vector)
        query_norm = math.sqrt(sum(v * v for v in query)) or 1.0

        scored: List[Tuple[float, _IndexedChunk]] = []
        for chunk in reports:
            if len(chunk.vector) != len(query):
                continue
            dot = sum(a * b for a, b in zip(query, chunk.vector))
            score = dot / (query_norm * chunk.norm)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                parent_doc_id="communities",
                score=round(score, 4),
                retrieval_path="community",
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]

    def stats(self, tenant_id: str) -> Dict[str, Any]:
        index = self._indexes.get(tenant_id)
        if index is None:
            return {"cached": False}
        return {
            "cached": True,
            "chunks": index.size,
            "embedding_version": index.embedding_version,
            "skipped_incompatible": index.skipped_incompatible,
            "age_seconds": round(index.age_seconds, 1),
        }


vector_index_service = VectorIndexService()
