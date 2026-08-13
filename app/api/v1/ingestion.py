"""Ingestion API Endpoint: POST /api/v1/ingestion/document."""
import time
from fastapi import APIRouter, Depends, status
from app.core.security import verify_api_key, verify_tenant_header
from app.models.payload import IngestionRequest, IngestionResponse
from app.services.chunking_service import chunking_service
from app.services.extraction_service import extraction_service
from app.services.resolution_service import resolution_service
from app.services.arcadedb_client import arcadedb_client

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])

@router.post(
    "/document",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text into tenant graph & vector index"
)
async def ingest_document(
    request: IngestionRequest,
    api_key: str = Depends(verify_api_key),
    x_tenant_id: str = Depends(verify_tenant_header)
) -> IngestionResponse:
    """Ingest, chunk, extract entities & relationships, resolve duplicates, and write to ArcadeDB."""
    t0 = time.perf_counter()
    tenant = x_tenant_id or request.tenant_id

    # 1. Chunk document
    chunks = chunking_service.chunk_document(doc_id=request.doc_id, content=request.content, metadata=request.metadata)

    # 2. Extract vertices and edges across all chunks
    all_vertices = []
    all_edges = []
    for chunk in chunks:
        v_list, e_list = extraction_service.extract_from_chunk(chunk.text, chunk.chunk_id)
        all_vertices.extend(v_list)
        all_edges.extend(e_list)

    # 3. Resolve and deduplicate entities
    resolved_vertices, resolved_edges = resolution_service.resolve_and_merge(all_vertices, all_edges)

    # 4. Insert into ArcadeDB tenant database
    for v in resolved_vertices:
        cypher = "CREATE (n:Entity {id: $id, label: $label, name: $name})"
        params = {"id": v.id, "label": v.label, "name": v.properties.get("name", v.id)}
        await arcadedb_client.execute_cypher(tenant, cypher, params)

    for e in resolved_edges:
        cypher = f"MATCH (a:Entity {{id: $src}}), (b:Entity {{id: $tgt}}) CREATE (a)-[:{e.type}]->(b)"
        params = {"src": e.source, "tgt": e.target}
        await arcadedb_client.execute_cypher(tenant, cypher, params)

    t1 = time.perf_counter()
    duration_ms = round((t1 - t0) * 1000, 2)

    return IngestionResponse(
        tenant_id=tenant,
        doc_id=request.doc_id,
        chunks_created=len(chunks),
        entities_extracted=len(resolved_vertices),
        relationships_created=len(resolved_edges),
        status="success",
        execution_time_ms=duration_ms
    )
