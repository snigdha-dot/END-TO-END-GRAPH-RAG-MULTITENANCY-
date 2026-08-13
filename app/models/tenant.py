"""Tenant metadata models."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class TenantSchemaConfig(BaseModel):
    tenant_id: str
    db_name: str
    allowed_vertex_labels: List[str] = Field(default_factory=lambda: ["Person", "Service", "Team", "Document", "Concept"])
    allowed_edge_types: List[str] = Field(default_factory=lambda: ["DEPENDS_ON", "MANAGES", "OWNS", "CITES", "HAS_PART"])
    created_at: str
