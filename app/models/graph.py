"""Graph primitives: Vertex, Edge, Subgraph, and Triples."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Vertex(BaseModel):
    id: str = Field(..., description="Unique vertex node ID")
    label: str = Field(..., description="Entity type/label e.g. Person, Service, Document")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node attributes")


class Edge(BaseModel):
    source: str = Field(..., description="Source vertex ID")
    target: str = Field(..., description="Target vertex ID")
    type: str = Field(..., description="Relationship type e.g. DEPENDS_ON, MANAGES, OWNS")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Edge metadata, confidence, weight")


class Subgraph(BaseModel):
    nodes: List[Vertex] = Field(default_factory=list, description="Retrieved vertices")
    edges: List[Edge] = Field(default_factory=list, description="Retrieved edges connecting vertices")


class Triple(BaseModel):
    source: str
    relation: str
    target: str
    confidence: float = 1.0
