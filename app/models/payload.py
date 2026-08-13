"""Pydantic V2 Request & Response schemas for Team B API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.graph import Subgraph

class RetrievalOptions(BaseModel):
    include_vector_search: bool = True
    max_traversal_depth: int = Field(default=2, ge=1, le=3)
    min_confidence_score: float = Field(default=0.75, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalRequest(BaseModel):
    tenant_id: str = Field(..., description="Target chatbot tenant database identifier")
    user_query: str = Field(..., min_length=2, description="Natural language search query")
    options: RetrievalOptions = Field(default_factory=RetrievalOptions)


class RetrievalResponse(BaseModel):
    tenant_id: str
    query: str
    subgraph: Subgraph
    context_passages: List[str]
    telemetry: Dict[str, Any] = Field(..., description="Side-by-side Latency (ms) and Price ($) metrics")


class IngestionRequest(BaseModel):
    tenant_id: str = Field(..., description="Target tenant database")
    doc_id: str = Field(..., description="Unique document ID")
    content: str = Field(..., min_length=10, description="Raw text content to chunk, extract, and index")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class IngestionResponse(BaseModel):
    tenant_id: str
    doc_id: str
    chunks_created: int
    entities_extracted: int
    relationships_created: int
    status: str = "success"
    execution_time_ms: float
