"""Graph primitives: Vertex, Edge, Subgraph, Triple, and retrieved chunks."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Vertex(BaseModel):
    id: str = Field(..., description="Canonical entity identifier")
    label: str = Field(..., description="Schema-approved vertex label, e.g. Film, Model")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node attributes")

    @property
    def name(self) -> str:
        return str(self.properties.get("name", self.id))


class Edge(BaseModel):
    source: str = Field(..., description="Source entity id")
    target: str = Field(..., description="Target entity id")
    type: str = Field(..., description="Schema-approved relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Confidence, provenance")

    @property
    def confidence(self) -> float:
        return float(self.properties.get("confidence", 1.0))


class Subgraph(BaseModel):
    nodes: List[Vertex] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.nodes and not self.edges


class Triple(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float = 1.0


class RetrievedChunk(BaseModel):
    """A text chunk returned by vector search, with its provenance and score."""

    chunk_id: str
    text: str
    parent_doc_id: str
    score: float = 0.0
    section_path: List[str] = Field(default_factory=list)
    retrieval_path: str = Field(
        default="vector", description="Which path surfaced this: vector, graph, or fused"
    )
    rank: Optional[int] = None
    rrf_score: float = Field(default=0.0, description="Reciprocal Rank Fusion score")


class LinkedEntity(BaseModel):
    """A query mention resolved to a canonical graph entity."""

    mention: str
    entity_id: str
    name: str
    label: str
    score: float
    method: str = Field(description="How it was linked: vector_knn, string_match, or exact")
