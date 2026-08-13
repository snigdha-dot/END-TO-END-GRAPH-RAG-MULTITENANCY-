"""Tenant metadata models."""
from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class TenantSchemaConfig(BaseModel):
    """Serializable view of a tenant's approved graph schema."""

    tenant_id: str
    db_name: str
    domain: str = "generic"
    display_name: str = ""
    allowed_vertex_labels: List[str] = Field(default_factory=list)
    allowed_edge_types: List[str] = Field(default_factory=list)
    default_traversal_edges: List[str] = Field(default_factory=list)
    created_at: str = ""


class TenantProvisionResult(BaseModel):
    tenant_id: str
    database: str
    vertex_types: List[str] = Field(default_factory=list)
    edge_types: List[str] = Field(default_factory=list)
    indexes: List[str] = Field(default_factory=list)
    created_at: str = ""
    details: Dict[str, object] = Field(default_factory=dict)
