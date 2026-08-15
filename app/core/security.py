"""Security Layer 3: query parameterization, input guarding, and token verification.

Two rules hold throughout:
  1. User-supplied values are *always* bound parameters, never interpolated.
  2. The only strings ever interpolated into a query are graph identifiers that
     were validated against the tenant's approved schema AND match a strict
     identifier pattern. Both checks, every time.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.exceptions import AuthenticationError, SecurityViolationError
from app.core.tenant_schema import TenantGraphSchema, is_safe_identifier


class CypherParameterizer:
    """Builds strictly parameterized, depth-bounded Cypher for graph traversal."""

    # Patterns that have no legitimate place in a natural-language user query.
    DANGEROUS_PATTERNS: Tuple[Tuple[str, str], ...] = (
        (r";\s*(?:DROP|DELETE|DETACH|CREATE|MERGE|SET|REMOVE|ALTER|TRUNCATE)\b", "statement chaining"),
        (r"\b(?:DROP|TRUNCATE)\s+(?:DATABASE|TYPE|VERTEX|EDGE|INDEX|BUCKET)\b", "schema destruction"),
        (r"\bDETACH\s+DELETE\b", "destructive delete"),
        (r"\bLOAD\s+CSV\b", "external data load"),
        (r"\bUNION\s+(?:ALL\s+)?MATCH\b", "query union injection"),
        (r"\bCALL\s+(?:db|dbms|apoc)\b", "procedure invocation"),
        (r"/\*|\*/", "block comment"),
        (r"(?:^|\s)--(?:\s|$)", "line comment"),
        (r"\$\{", "template interpolation"),
        (r"\bJAVASCRIPT\b|\bSCRIPT\s*:", "script execution"),
    )

    MAX_QUERY_LENGTH = 2000

    @classmethod
    def guard_user_text(cls, text: str, field_name: str = "input") -> str:
        """Reject text carrying injection signatures; return the trimmed value.

        This is defence in depth, not the primary control — the primary control is
        that this text is bound as a parameter and never concatenated.
        """
        if text is None:
            return ""
        if len(text) > cls.MAX_QUERY_LENGTH:
            raise SecurityViolationError(
                f"{field_name} exceeds the maximum length of {cls.MAX_QUERY_LENGTH} characters.",
                field=field_name,
            )
        if "\x00" in text:
            raise SecurityViolationError(f"{field_name} contains a null byte.", field=field_name)
        for pattern, description in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise SecurityViolationError(
                    f"Potentially malicious pattern detected in {field_name}: {description}.",
                    field=field_name,
                    violation=description,
                )
        return text.strip()

    # Kept as an alias so existing callers/tests keep working.
    @classmethod
    def sanitize_input(cls, text: str) -> str:
        return cls.guard_user_text(text)

    @classmethod
    def safe_edge_fragment(cls, edge_types: List[str], schema: TenantGraphSchema) -> str:
        """Build a `:A|:B` relationship filter from schema-approved identifiers only."""
        approved: List[str] = []
        for etype in edge_types:
            if schema.validate_edge_type(etype) and is_safe_identifier(etype):
                approved.append(etype)
        if not approved:
            return ""
        return ":" + "|:".join(approved)

    @classmethod
    def build_parameterized_traversal(
        cls,
        start_node_ids: List[str],
        rel_types: Optional[List[str]] = None,
        max_depth: int = 2,
        schema: Optional[TenantGraphSchema] = None,
        limit: Optional[int] = None,
        seed_label: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Construct a bounded, parameterized multi-hop traversal query.

        Node IDs and the result limit are bound parameters. Only the depth bound and
        the schema-validated relationship labels appear in the query text, because
        Cypher does not permit either as a parameter.
        """
        if schema is None:
            from app.core.tenant_context import get_tenant_context

            schema = get_tenant_context().schema

        # Hard bounds (plan section 2.2) regardless of what the caller requested.
        safe_depth = min(max(1, int(max_depth)), settings.MAX_TRAVERSAL_DEPTH)
        safe_limit = min(max(1, int(limit or settings.MAX_TRAVERSAL_NODES)), settings.MAX_TRAVERSAL_NODES)

        requested = rel_types if rel_types is not None else schema.traversal_edges()
        rel_fragment = cls.safe_edge_fragment(requested, schema)

        # Naming the start label is a correctness-of-performance requirement, not a
        # style choice: an untyped `(start)` makes ArcadeDB scan every vertex type
        # instead of using the UNIQUE index on entity_id. Measured at 61,901ms for
        # a depth-2 traversal versus 35ms labelled, on the same 400-chunk tenant.
        #
        # The label is schema-validated before interpolation; an unrecognised one
        # falls back to the untyped form, which is slow but still correct.
        start_label = ""
        if seed_label and schema.validate_vertex_label(seed_label) and is_safe_identifier(seed_label):
            start_label = f":{seed_label}"

        # ArcadeDB's Cypher layer does not implement the path functions
        # `nodes(path)` / `relationships(path)`, so endpoints and the relationship
        # type are projected directly. Variable-length matching itself is supported,
        # so multi-hop traversal is unaffected.
        cypher = (
            f"MATCH (start{start_label})-[rel{rel_fragment}*1..{safe_depth}]-(related) "
            "WHERE start.entity_id IN $start_nodes "
            "RETURN start.entity_id AS source_id, start.name AS source_name, "
            "start.entity_label AS source_label, "
            "related.entity_id AS target_id, related.name AS target_name, "
            "related.entity_label AS target_label "
            "LIMIT $limit"
        )
        params: Dict[str, Any] = {"start_nodes": list(start_node_ids), "limit": safe_limit}
        return cypher, params

    @classmethod
    def build_entity_candidate_lookup(cls, names: List[str], limit: int = 25) -> Tuple[str, Dict[str, Any]]:
        """Look up canonical entities by normalized name (fully parameterized).

        Alias matching is deliberately not expressed here: ArcadeDB's Cypher layer
        rejects `ANY(a IN e.aliases WHERE ...)` list predicates. Aliases are returned
        with each candidate and matched in `resolution_service`, which also applies
        Jaro-Winkler and vector scoring — so recall is unaffected.
        """
        cypher = (
            "MATCH (e) "
            "WHERE e.normalized_name IN $names "
            "RETURN e.entity_id AS entity_id, e.name AS name, e.entity_label AS label, "
            "e.normalized_name AS normalized_name, e.aliases AS aliases "
            "LIMIT $limit"
        )
        return cypher, {"names": list(names), "limit": int(limit)}


class TenantIdValidator:
    """Validation for tenant identifiers used to build database names."""

    PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,62}$")

    @classmethod
    def validate(cls, raw: str) -> str:
        """Normalize and validate a tenant id, rejecting anything unsafe.

        Rejects rather than strips: silently rewriting `movies-bot!` into `moviesbot`
        would route a typo to the wrong (or a newly created) knowledge base.
        """
        if not raw or not raw.strip():
            raise SecurityViolationError("Tenant identifier is missing or empty.", field="tenant_id")
        candidate = raw.strip().lower()
        if not cls.PATTERN.match(candidate):
            raise SecurityViolationError(
                "Tenant identifier must be lowercase alphanumeric with underscores, "
                "2-63 characters, starting with a letter.",
                field="tenant_id",
            )
        return candidate


class JWTVerifier:
    """Minimal, dependency-free HS256 JWT verification.

    Implemented in-tree so the service has no hard dependency on PyJWT for a
    single algorithm. Verifies signature, `exp`, `nbf`, `iss`, `aud`, and returns
    the claim set; `tenant_id` is then read from the *verified* claims, never from
    a client-controlled header.
    """

    @staticmethod
    def _b64url_decode(segment: str) -> bytes:
        import base64

        padding = "=" * (-len(segment) % 4)
        return base64.urlsafe_b64decode(segment + padding)

    @classmethod
    def verify(cls, token: str) -> Dict[str, Any]:
        import json

        if not token or token.count(".") != 2:
            raise AuthenticationError("Malformed JWT: expected three dot-separated segments.")

        header_b64, payload_b64, signature_b64 = token.split(".")

        try:
            header = json.loads(cls._b64url_decode(header_b64))
            claims = json.loads(cls._b64url_decode(payload_b64))
            signature = cls._b64url_decode(signature_b64)
        except Exception as exc:
            raise AuthenticationError("Malformed JWT: segments are not valid base64url JSON.") from exc

        alg = header.get("alg")
        if alg != settings.JWT_ALGORITHM:
            # Blocks the `alg: none` downgrade and algorithm-confusion attacks.
            raise AuthenticationError(f"Unsupported JWT algorithm: {alg!r}.")

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected = hmac.new(
            settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(expected, signature):
            raise AuthenticationError("JWT signature verification failed.")

        now = int(time.time())
        leeway = settings.JWT_LEEWAY_SECONDS

        exp = claims.get("exp")
        if exp is None:
            raise AuthenticationError("JWT is missing the required 'exp' claim.")
        if now > int(exp) + leeway:
            raise AuthenticationError("JWT has expired.")

        nbf = claims.get("nbf")
        if nbf is not None and now + leeway < int(nbf):
            raise AuthenticationError("JWT is not yet valid.")

        if settings.JWT_ISSUER and claims.get("iss") != settings.JWT_ISSUER:
            raise AuthenticationError("JWT issuer mismatch.")

        if settings.JWT_AUDIENCE:
            aud = claims.get("aud")
            audiences = aud if isinstance(aud, list) else [aud]
            if settings.JWT_AUDIENCE not in audiences:
                raise AuthenticationError("JWT audience mismatch.")

        if not claims.get("tenant_id"):
            raise AuthenticationError("JWT is missing the required 'tenant_id' claim.")

        return claims

    @classmethod
    def issue(cls, tenant_id: str, user_id: Optional[str] = None, ttl_seconds: int = 900,
              scopes: Optional[List[str]] = None) -> str:
        """Issue a token. Used by tests and local tooling, not by the request path."""
        import base64
        import json

        def b64(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

        now = int(time.time())
        header = {"alg": settings.JWT_ALGORITHM, "typ": "JWT"}
        claims: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "iss": settings.JWT_ISSUER,
            "aud": settings.JWT_AUDIENCE,
            "iat": now,
            "nbf": now,
            "exp": now + ttl_seconds,
            "scope": scopes or ["retrieval:read"],
        }
        if user_id:
            claims["user_id"] = user_id

        header_b64 = b64(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = b64(json.dumps(claims, separators=(",", ":")).encode())
        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        signature = hmac.new(
            settings.JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        return f"{header_b64}.{payload_b64}.{b64(signature)}"


def api_key_fingerprint(api_key: str) -> str:
    """Stable, non-reversible key identifier for audit logs."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
