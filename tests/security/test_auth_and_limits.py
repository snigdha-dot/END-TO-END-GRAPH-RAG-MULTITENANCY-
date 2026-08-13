"""JWT verification, rate limiting, and response-hardening tests."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.security import JWTVerifier, api_key_fingerprint


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


@pytest.fixture(autouse=True)
def _enable_jwt(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "JWT_ENABLED", True, raising=False)
    monkeypatch.setattr(
        settings, "JWT_SECRET", "test_secret_at_least_32_characters_long!!", raising=False
    )
    yield


# --------------------------------------------------------------- JWT round trip
def test_valid_token_verifies_and_carries_tenant():
    token = JWTVerifier.issue("movies_bot", user_id="u_1")
    claims = JWTVerifier.verify(token)
    assert claims["tenant_id"] == "movies_bot"
    assert claims["user_id"] == "u_1"


def test_expired_token_is_rejected():
    token = JWTVerifier.issue("movies_bot", ttl_seconds=-3600)
    with pytest.raises(AuthenticationError, match="expired"):
        JWTVerifier.verify(token)


def test_tampered_payload_fails_signature_check():
    """Forging the tenant claim must not work: that is the whole point of signing."""
    token = JWTVerifier.issue("movies_bot")
    header_b64, payload_b64, signature = token.split(".")

    claims = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))
    claims["tenant_id"] = "ai_trends_bot"
    forged_payload = _b64(json.dumps(claims, separators=(",", ":")).encode())

    with pytest.raises(AuthenticationError, match="signature"):
        JWTVerifier.verify(f"{header_b64}.{forged_payload}.{signature}")


def test_alg_none_downgrade_is_rejected():
    """The classic JWT attack: strip the algorithm and supply no signature."""
    header = _b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    claims = _b64(
        json.dumps(
            {
                "tenant_id": "ai_trends_bot",
                "exp": int(time.time()) + 3600,
                "iss": settings.JWT_ISSUER,
                "aud": settings.JWT_AUDIENCE,
            }
        ).encode()
    )
    with pytest.raises(AuthenticationError, match="algorithm"):
        JWTVerifier.verify(f"{header}.{claims}.")


def test_token_signed_with_wrong_secret_is_rejected():
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    claims = _b64(
        json.dumps(
            {
                "tenant_id": "ai_trends_bot",
                "exp": int(time.time()) + 3600,
                "iss": settings.JWT_ISSUER,
                "aud": settings.JWT_AUDIENCE,
            }
        ).encode()
    )
    forged = hmac.new(b"attacker_secret", f"{header}.{claims}".encode(), hashlib.sha256).digest()
    with pytest.raises(AuthenticationError):
        JWTVerifier.verify(f"{header}.{claims}.{_b64(forged)}")


def test_missing_tenant_claim_is_rejected():
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    claims = _b64(
        json.dumps(
            {"exp": int(time.time()) + 3600, "iss": settings.JWT_ISSUER,
             "aud": settings.JWT_AUDIENCE}
        ).encode()
    )
    signature = hmac.new(
        settings.JWT_SECRET.encode(), f"{header}.{claims}".encode(), hashlib.sha256
    ).digest()
    with pytest.raises(AuthenticationError, match="tenant_id"):
        JWTVerifier.verify(f"{header}.{claims}.{_b64(signature)}")


@pytest.mark.parametrize("malformed", ["", "not.a.jwt.x", "onlyonepart", "two.parts"])
def test_malformed_tokens_are_rejected(malformed):
    with pytest.raises(AuthenticationError):
        JWTVerifier.verify(malformed)


def test_jwt_tenant_must_match_api_key_tenant(client, movies_headers):
    """A token for one chatbot replayed with another chatbot's key must fail."""
    token = JWTVerifier.issue("ai_trends_bot")
    response = client.post(
        "/api/v1/retrieval/search",
        headers={**movies_headers, "Authorization": f"Bearer {token}"},
        json={"user_query": "anything at all"},
    )
    assert response.status_code == 403


# --------------------------------------------------------------- key handling
def test_fingerprint_is_stable_and_non_reversible():
    fp = api_key_fingerprint("secret_key_value")
    assert fp == api_key_fingerprint("secret_key_value")
    assert "secret_key_value" not in fp
    assert len(fp) == 12


# --------------------------------------------------------------- HTTP hardening
def test_security_headers_are_applied(client):
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Content-Security-Policy" in response.headers


def test_request_id_is_returned(client):
    response = client.get("/health")
    assert response.headers.get(settings.REQUEST_ID_HEADER)


def test_rate_limit_returns_429(client, movies_headers, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 3, raising=False)
    monkeypatch.setattr(settings, "RATE_LIMIT_BURST", 0, raising=False)

    statuses = [
        client.post(
            "/api/v1/retrieval/search", headers=movies_headers,
            json={"user_query": "test query"},
        ).status_code
        for _ in range(6)
    ]
    assert 429 in statuses


def test_oversized_body_is_rejected(client, movies_headers):
    response = client.post(
        "/api/v1/ingest/document",
        headers={**movies_headers, "Content-Length": str(settings.MAX_REQUEST_BODY_BYTES + 1)},
        json={"doc_id": "d", "content": "x" * 100},
    )
    assert response.status_code in (413, 422)


def test_unknown_fields_are_rejected(client, movies_headers):
    """extra='forbid' stops silently-ignored typos in the client contract."""
    response = client.post(
        "/api/v1/retrieval/search",
        headers=movies_headers,
        json={"user_query": "valid query", "unexpected_field": "value"},
    )
    assert response.status_code == 422
