"""Security tests for Cypher Injection prevention and input sanitization."""
import pytest
from fastapi import HTTPException
from app.core.security import CypherParameterizer

def test_cypher_parameterization_safely_binds():
    nodes = ["canon_auth_service", "canon_payment"]
    rels = ["DEPENDS_ON"]

    cypher, params = CypherParameterizer.build_parameterized_traversal(nodes, rels, max_depth=2)

    assert "MATCH path =" in cypher
    assert "start.id IN $start_nodes" in cypher
    assert params["start_nodes"] == nodes
    assert params["limit"] > 0


def test_malicious_input_sanitization_raises_exception():
    malicious_input = "Auth Service'; DROP DATABASE tenant_tech_kb; --"

    with pytest.raises(HTTPException) as exc_info:
        CypherParameterizer.sanitize_input(malicious_input)

    assert exc_info.value.status_code == 400
    assert "Security Alert" in exc_info.value.detail
