"""Security, Cypher Parameterization, and Tenant Isolation Guard."""
import re
from typing import Any, Dict, Tuple
from fastapi import HTTPException, Security, Header, status
from app.core.config import settings

class CypherParameterizer:
    """Security utility to ensure Cypher queries are strictly parameterized and injection-free."""

    DANGEROUS_PATTERNS = [
        r";\s*DROP",
        r";\s*DELETE",
        r";\s*DETACH",
        r";\s*CREATE",
        r"--",
        r"/\*",
        r"\*/",
        r"UNION\s+ALL",
    ]

    @classmethod
    def sanitize_input(cls, text: str) -> str:
        """Sanitize raw text inputs to prevent injection attempts."""
        if not text:
            return ""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Security Alert: Potentially malicious pattern detected in input: {pattern}"
                )
        return text.strip()

    @classmethod
    def build_parameterized_traversal(
        cls, start_node_ids: list[str], rel_types: list[str], max_depth: int = 2
    ) -> Tuple[str, Dict[str, Any]]:
        """Construct a safe parameterized Cypher query with strict depth limits."""
        # Sanitize depth to hard upper bound of 3
        safe_depth = min(max(1, max_depth), 3)

        # Validate relationship types against alphanumeric + underscore pattern
        safe_rels = []
        for rel in rel_types:
            clean_rel = re.sub(r"[^a-zA-Z0-9_]", "", rel)
            if clean_rel:
                safe_rels.append(clean_rel)

        rel_filter = "|:".join(safe_rels) if safe_rels else ""
        rel_cypher = f":{rel_filter}" if rel_filter else ""

        # Parameterized query template
        cypher = (
            f"MATCH path = (start)-[{rel_cypher}*1..{safe_depth}]-(end) "
            f"WHERE start.id IN $start_nodes "
            f"RETURN nodes(path) as nodes, relationships(path) as edges "
            f"LIMIT $limit"
        )

        params = {
            "start_nodes": start_node_ids,
            "limit": settings.MAX_TRAVERSAL_NODES
        }

        return cypher, params


async def verify_api_key(x_api_key: str = Header(..., alias=settings.API_KEY_HEADER)) -> str:
    """Validate API key presented by Team A chatbot service."""
    if x_api_key not in settings.ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key."
        )
    return x_api_key


async def verify_tenant_header(x_tenant_id: str = Header(..., alias=settings.TENANT_HEADER)) -> str:
    """Ensure a tenant context header is explicitly attached to the request."""
    clean_tenant = re.sub(r"[^a-zA-Z0-9_-]", "", x_tenant_id)
    if not clean_tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or empty tenant ID format."
        )
    return clean_tenant
