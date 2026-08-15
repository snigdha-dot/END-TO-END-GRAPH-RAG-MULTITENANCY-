"""Ingestion orchestration: one pipeline for every source format.

    detect -> adapt -> chunk -> validate -> embed -> extract
           -> resolve -> build graph -> detect communities -> report

Replaces the two parallel pipelines that duplicated resolution, validation, and
the write path. Format handling now stops at the adapter, so everything after the
CanonicalDocument is format-agnostic.

Documents are processed in windows of chunks. Holding every chunk, vector, and
entity for a large source in memory before the first write does not survive a real
dataset.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.core.telemetry import TelemetryTracker
from app.core.tenant_context import TenantContext
from app.models.canonical import CanonicalChunk, CanonicalDocument
from app.models.graph import Edge, Vertex
from app.services.adapters import adapter_registry
from app.services.arcadedb_client import arcadedb_client
from app.services.canonical_chunker import canonical_chunker
from app.services.community_service import Community, community_service
from app.services.embedding_service import embedding_service
from app.services.extraction_router import extraction_router
from app.services.graph_builder import GraphWriteResult, graph_builder
from app.services.resolution_service import resolution_service

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    """Everything the pipeline did, for the caller and for telemetry."""

    tenant_id: str = ""
    doc_id: str = ""
    source_format: str = ""
    blocks: int = 0
    chunks_produced: int = 0
    chunks_rejected: int = 0
    chunk_kinds: Dict[str, int] = field(default_factory=dict)
    entities_extracted: int = 0
    entities_written: int = 0
    edges_extracted: int = 0
    edges_written: int = 0
    mentions_written: int = 0
    communities_built: int = 0
    statements_executed: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    extraction_methods: Dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def merge_write(self, result: GraphWriteResult) -> None:
        self.entities_written += result.entities_written
        self.edges_written += result.edges_written
        self.mentions_written += result.mentions_written
        self.statements_executed += result.statements_executed
        for reason, count in result.rejection_reasons.items():
            self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "doc_id": self.doc_id,
            "source_format": self.source_format,
            "blocks": self.blocks,
            "chunks_created": self.chunks_produced,
            "chunks_rejected": self.chunks_rejected,
            "chunk_kinds": self.chunk_kinds,
            "entities_extracted": self.entities_extracted,
            "entities_written": self.entities_written,
            "relationships_created": self.edges_written,
            "edges_extracted": self.edges_extracted,
            "mentions_linked": self.mentions_written,
            "communities_built": self.communities_built,
            "statements_executed": self.statements_executed,
            "rejection_reasons": self.rejection_reasons,
            "extraction_methods": self.extraction_methods,
            "llm_tokens": {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
            },
        }


class IngestionPipeline:
    """Runs a source document end to end into a tenant knowledge graph."""

    CHUNK_WINDOW = 100

    async def ingest_file(
        self,
        ctx: TenantContext,
        source_path: Any,
        max_rows: Optional[int] = None,
        subject_column: Optional[str] = None,
        build_communities: bool = True,
    ) -> Dict[str, Any]:
        """Ingest any supported file into the tenant's graph."""
        telemetry = TelemetryTracker()
        path = Path(source_path)

        t0 = time.perf_counter()
        document = adapter_registry.to_canonical(
            path, max_rows=max_rows, subject_column=subject_column
        )
        telemetry.record_step_latency("format_adaptation", (time.perf_counter() - t0) * 1000)

        return await self._run(ctx, document, telemetry, build_communities)

    async def ingest_text(
        self,
        ctx: TenantContext,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        build_communities: bool = False,
    ) -> Dict[str, Any]:
        """Ingest raw text, for callers that already hold the content."""
        telemetry = TelemetryTracker()

        t0 = time.perf_counter()
        virtual_path = Path(f"{doc_id}.md")
        document = adapter_registry.prose_parser.to_document(
            virtual_path, text=content, source_format="text"
        )
        document.doc_id = doc_id
        if metadata:
            document.metadata.update(metadata)
        telemetry.record_step_latency("format_adaptation", (time.perf_counter() - t0) * 1000)

        return await self._run(ctx, document, telemetry, build_communities)

    # ------------------------------------------------------------------ core
    async def _run(
        self,
        ctx: TenantContext,
        document: CanonicalDocument,
        telemetry: TelemetryTracker,
        build_communities: bool,
    ) -> Dict[str, Any]:
        schema = ctx.schema
        report = IngestionReport(
            tenant_id=ctx.tenant_id,
            doc_id=document.doc_id,
            source_format=document.provenance.source_format,
            blocks=len(document.blocks),
        )

        if document.is_empty:
            telemetry.record_step_latency("chunking", 0.0)
            return {**report.to_dict(), "status": "empty", "telemetry": telemetry.finalize()}

        # ---------------------------------------------------------- chunk
        t0 = time.perf_counter()
        all_chunks = canonical_chunker.chunk(document)
        valid_chunks = [c for c in all_chunks if c.is_valid]
        telemetry.record_step_latency("chunking", (time.perf_counter() - t0) * 1000)

        report.chunks_produced = len(valid_chunks)
        report.chunks_rejected = len(all_chunks) - len(valid_chunks)
        for chunk in valid_chunks:
            report.chunk_kinds[chunk.kind.value] = report.chunk_kinds.get(chunk.kind.value, 0) + 1

        if not valid_chunks:
            return {**report.to_dict(), "status": "no_valid_chunks", "telemetry": telemetry.finalize()}

        # Entities accumulate across windows so communities see the whole document.
        document_entities: List[Vertex] = []
        document_edges: List[Edge] = []
        embed_ms = resolve_ms = extract_ms = write_ms = 0.0

        for start in range(0, len(valid_chunks), self.CHUNK_WINDOW):
            window = valid_chunks[start : start + self.CHUNK_WINDOW]

            # ------------------------------------------------------ embed
            t1 = time.perf_counter()
            vectors = embedding_service.encode_batch([c.embedding_text() for c in window])
            embed_ms += (time.perf_counter() - t1) * 1000

            # ------------------------------------------------------ extract
            t2 = time.perf_counter()
            entities, edges, mentions, stats = await extraction_router.extract_many(
                window, schema
            )
            extract_ms += (time.perf_counter() - t2) * 1000

            report.entities_extracted += len(entities)
            report.edges_extracted += len(edges)
            report.prompt_tokens += stats["prompt_tokens"]
            report.completion_tokens += stats["completion_tokens"]
            for method, count in stats["methods"].items():
                report.extraction_methods[method] = (
                    report.extraction_methods.get(method, 0) + count
                )

            # ------------------------------------------------------ resolve
            t3 = time.perf_counter()
            resolved_entities, resolved_edges = resolution_service.resolve_and_merge(
                entities, edges, schema=schema, use_embeddings=False
            )
            resolve_ms += (time.perf_counter() - t3) * 1000

            canonical_ids = {v.id for v in resolved_entities}
            resolved_mentions = {
                (eid, cid) for eid, cid in mentions if eid in canonical_ids
            }

            # ------------------------------------------------------ write
            t4 = time.perf_counter()
            write_result = await graph_builder.write(
                tenant_id=ctx.tenant_id,
                schema=schema,
                chunks=window,
                vectors=vectors,
                entities=resolved_entities,
                edges=resolved_edges,
                mentions=resolved_mentions,
            )
            write_ms += (time.perf_counter() - t4) * 1000
            report.merge_write(write_result)

            document_entities.extend(resolved_entities)
            document_edges.extend(resolved_edges)

        telemetry.record_step_latency("chunk_embedding", embed_ms)
        telemetry.record_step_latency("entity_extraction", extract_ms)
        telemetry.record_step_latency("entity_resolution", resolve_ms)
        telemetry.record_step_latency("graph_write", write_ms)
        telemetry.record_model_call(
            step_name="chunk_embedding",
            model_name=embedding_service.model_label,
            prompt_tokens=sum(c.token_count for c in valid_chunks),
            completion_tokens=0,
            duration_ms=embed_ms,
        )
        if report.prompt_tokens or report.completion_tokens:
            telemetry.record_model_call(
                step_name="entity_extraction",
                model_name=settings.LLM_MODEL,
                prompt_tokens=report.prompt_tokens,
                completion_tokens=report.completion_tokens,
                duration_ms=extract_ms,
            )

        # ---------------------------------------------------------- communities
        if build_communities and document_entities:
            t5 = time.perf_counter()
            report.communities_built = await self._build_communities(
                ctx, document_entities, document_edges, schema
            )
            telemetry.record_step_latency(
                "community_detection", (time.perf_counter() - t5) * 1000
            )

        result = telemetry.finalize()
        return {
            **report.to_dict(),
            "embedding_model": embedding_service.model_label,
            "extraction_backend": extraction_router.prose_method,
            "llm_extraction": extraction_router.llm_available,
            "status": "success",
            "telemetry": result,
            "execution_time_ms": result["latency_breakdown_ms"]["total_retrieval_latency"],
        }

    async def _build_communities(
        self,
        ctx: TenantContext,
        entities: Sequence[Vertex],
        edges: Sequence[Edge],
        schema,
    ) -> int:
        """Detect communities, write their reports as searchable chunks."""
        # Deduplicate: a window boundary can yield the same canonical entity twice.
        unique_entities = list({v.id: v for v in entities}.values())
        unique_edges = list({(e.source, e.type, e.target): e for e in edges}.values())

        communities = await community_service.build(
            unique_entities, unique_edges, schema
        )
        if not communities:
            return 0

        vectors = embedding_service.encode_batch([c.report_text() for c in communities])
        statements = community_service.to_chunk_statements(communities, vectors)
        if statements:
            await arcadedb_client.execute_batch(statements, tenant_id=ctx.tenant_id)

        return len(communities)


ingestion_pipeline = IngestionPipeline()
