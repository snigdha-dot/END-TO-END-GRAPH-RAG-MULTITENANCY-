"""Multi-tenant isolation tests (plan section 7, row 1).

The previous version of this file only checked that a tenant header matched a
regex. It never attempted cross-tenant access, so the isolation guarantee was
asserted but never verified. These tests attempt the actual attack.
"""
from __future__ import annotations

import pytest

from app.api.dependencies import _resolve_api_key_tenant
from app.core.exceptions import (
    AuthenticationError,
    SecurityViolationError,
    TenantAccessDeniedError,
)
from app.core.security import TenantIdValidator
from app.core.tenant_context import TenantContext, get_tenant_context, tenant_scope
from tests.conftest import AI_TRENDS_KEY, MOVIES_KEY


# --------------------------------------------------------------- key -> tenant
def test_api_key_resolves_to_its_own_tenant():
    assert _resolve_api_key_tenant(MOVIES_KEY) == "movies_bot"
    assert _resolve_api_key_tenant(AI_TRENDS_KEY) == "ai_trends_bot"


def test_unknown_api_key_is_rejected():
    with pytest.raises(AuthenticationError):
        _resolve_api_key_tenant("not_a_real_key")


# --------------------------------------------------------------- HTTP boundary
def test_movies_key_cannot_read_ai_trends_tenant(client, movies_headers):
    """The core leakage test: a valid key asserting another tenant must be denied.

    Before key->tenant binding this returned 200 with the other tenant's data.
    """
    headers = {**movies_headers, "X-Tenant-ID": "ai_trends_bot"}
    response = client.post(
        "/api/v1/retrieval/search",
        headers=headers,
        json={"user_query": "What is a diffusion model?"},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "tenant_access_denied"


def test_ai_trends_key_cannot_read_movies_tenant(client, ai_trends_headers):
    headers = {**ai_trends_headers, "X-Tenant-ID": "movies_bot"}
    response = client.post(
        "/api/v1/retrieval/search",
        headers=headers,
        json={"user_query": "Who directed Inception?"},
    )
    assert response.status_code == 403


def test_payload_tenant_id_cannot_override_credential(client, movies_headers):
    """A tenant_id in the body must not redirect the query to another database."""
    response = client.post(
        "/api/v1/retrieval/search",
        headers=movies_headers,
        json={"tenant_id": "ai_trends_bot", "user_query": "diffusion models"},
    )
    # Either the tenant is ignored entirely (its own KB is queried) or the request
    # fails - but it must never be served against ai_trends_bot.
    assert response.status_code != 200 or response.json()["tenant_id"] == "movies_bot"


def test_missing_api_key_is_unauthenticated(client):
    response = client.post(
        "/api/v1/retrieval/search",
        headers={"Content-Type": "application/json"},
        json={"user_query": "anything"},
    )
    assert response.status_code == 401


def test_ingestion_enforces_the_same_binding(client, movies_headers):
    headers = {**movies_headers, "X-Tenant-ID": "ai_trends_bot"}
    response = client.post(
        "/api/v1/ingest/document",
        headers=headers,
        json={"doc_id": "d1", "content": "Some content long enough to pass validation."},
    )
    assert response.status_code == 403


def test_admin_provisioning_rejects_a_tenant_api_key(client, movies_headers):
    """A chatbot credential must not be able to create databases."""
    response = client.post(
        "/api/v1/tenant/provision",
        headers={"X-API-Key": MOVIES_KEY, "Content-Type": "application/json"},
        json={"tenant_id": "attacker_bot"},
    )
    assert response.status_code in (401, 403)


# --------------------------------------------------------------- tenant id rules
@pytest.mark.parametrize(
    "malicious",
    [
        "../../etc/passwd",
        "movies_bot/../ai_trends_bot",
        "movies_bot; DROP DATABASE",
        "movies bot",
        "movies-bot!",
        "1movies_bot",
        "!@#$%^&*()",
        "",
        "a" * 100,
    ],
)
def test_malicious_tenant_ids_are_rejected_not_sanitized(malicious):
    """Rejecting matters more than stripping.

    Silently rewriting `movies-bot!` into `moviesbot` would route the request to a
    different (or newly created) knowledge base instead of failing.
    """
    with pytest.raises(SecurityViolationError):
        TenantIdValidator.validate(malicious)


def test_valid_tenant_ids_are_accepted():
    assert TenantIdValidator.validate("movies_bot") == "movies_bot"
    assert TenantIdValidator.validate("  AI_Trends_Bot  ") == "ai_trends_bot"


# --------------------------------------------------------------- context guard
def test_unscoped_access_fails_closed():
    """Security Layer 2: no bound context means no database access."""
    with pytest.raises(TenantAccessDeniedError):
        get_tenant_context()


def test_tenant_scope_binds_and_releases():
    ctx = TenantContext(tenant_id="movies_bot", api_key_id="k", request_id="r")
    with tenant_scope(ctx):
        assert get_tenant_context().tenant_id == "movies_bot"
    with pytest.raises(TenantAccessDeniedError):
        get_tenant_context()


def test_tenant_context_maps_to_isolated_database():
    movies = TenantContext(tenant_id="movies_bot", api_key_id="k", request_id="r")
    ai = TenantContext(tenant_id="ai_trends_bot", api_key_id="k", request_id="r")
    assert movies.db_name == "tenant_movies_bot_kb"
    assert ai.db_name == "tenant_ai_trends_bot_kb"
    assert movies.db_name != ai.db_name


def test_tenant_schemas_share_no_vocabulary():
    """Domain isolation: a movies edge type must not be valid for AI trends."""
    movies = TenantContext(tenant_id="movies_bot", api_key_id="k", request_id="r").schema
    ai = TenantContext(tenant_id="ai_trends_bot", api_key_id="k", request_id="r").schema

    assert movies.validate_edge_type("DIRECTED")
    assert not ai.validate_edge_type("DIRECTED")
    assert ai.validate_edge_type("BUILDS_ON")
    assert not movies.validate_edge_type("BUILDS_ON")

    assert movies.validate_vertex_label("Film")
    assert not ai.validate_vertex_label("Film")
