"""Pydantic v2 request & response schemas for the Team B API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.graph import LinkedEntity, RetrievedChunk, Subgraph


class RetrievalOptions(BaseModel):
    include_vector_search: bool = True
    max_traversal_depth: int = Field(default=2, ge=1, le=3)
    min_confidence_score: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)
    include_subgraph: bool = True


class RetrievalRequest(BaseModel):
    """Search request from a Team A chatbot.

    `tenant_id` is accepted for backward compatibility but is NOT authoritative:
    the tenant is derived from the credential. A value disagreeing with the
    authenticated tenant is rejected by the auth layer rather than honoured.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: Optional[str] = Field(
        default=None,
        deprecated=True,
        description="Ignored for routing; the API key determines the tenant.",
    )
    user_query: str = Field(..., min_length=2, max_length=2000)
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)

    @field_validator("user_query")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user_query must not be blank.")
        return v


class RetrievalResponse(BaseModel):
    tenant_id: str
    query: str
    subgraph: Subgraph
    context_passages: List[str] = Field(
        description="Ready-to-inject context for the caller's LLM prompt"
    )
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    linked_entities: List[LinkedEntity] = Field(default_factory=list)
    telemetry: Dict[str, Any] = Field(
        description="Side-by-side latency (ms) and model cost (USD) metrics"
    )
    request_id: Optional[str] = None


class IngestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: Optional[str] = Field(default=None, deprecated=True)
    doc_id: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10, max_length=1_000_000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    tenant_id: str
    doc_id: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    mentions_linked: int = 0
    schema_rejections: int = 0
    statements_executed: int = 0
    embedding_model: str = ""
    extraction_backend: str = ""
    status: str = "success"
    execution_time_ms: float
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class TenantProvisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(..., min_length=2, max_length=63)


class TenantSchemaResponse(BaseModel):
    tenant_id: str
    db_name: str
    domain: str
    display_name: str
    allowed_vertex_labels: List[str]
    allowed_edge_types: List[str]
    default_traversal_edges: List[str]
    ner_labels: List[str] = Field(default_factory=list)
    provisioned: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    ready: bool
    service: str
    version: str
    arcadedb_ready: bool
    embedding_model: str
    semantic_embeddings: bool
    cross_encoder_active: bool
    extraction_backend: str
    configuration_warnings: List[str] = Field(default_factory=list)
