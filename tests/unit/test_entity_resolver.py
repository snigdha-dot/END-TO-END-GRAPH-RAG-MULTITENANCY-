"""Unit tests for entity disambiguation and merging service."""
import pytest
from app.models.graph import Vertex, Edge
from app.services.resolution_service import resolution_service

def test_entity_resolution_merges_similar_names():
    v1 = Vertex(id="v1", label="Service", properties={"name": "Auth Service"})
    v2 = Vertex(id="v2", label="Service", properties={"name": "auth_service"})
    v3 = Vertex(id="v3", label="Service", properties={"name": "Payment Gateway"})

    e1 = Edge(source="v1", target="v3", type="DEPENDS_ON", properties={})
    e2 = Edge(source="v2", target="v3", type="DEPENDS_ON", properties={})

    resolved_vertices, resolved_edges = resolution_service.resolve_and_merge([v1, v2, v3], [e1, e2])

    # Should merge "Auth Service" and "auth_service" into 1 canonical node
    assert len(resolved_vertices) == 2
    assert len(resolved_edges) == 1
    assert resolved_edges[0].source == "canon_auth_service"
