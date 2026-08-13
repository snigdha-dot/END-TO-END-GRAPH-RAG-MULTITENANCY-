"""Unit tests for semantic chunking service."""
import pytest
from app.services.chunking_service import chunking_service

def test_chunking_document():
    content = """
    # Section 1: Auth Architecture
    Auth Service manages OAuth2 tokens and handles user login credentials.
    Payment Service depends on Auth Service for validating API requests.

    # Section 2: User Management
    User Portal allows customers to view profile information and update billing details.
    User Portal depends on Auth Service as well.
    """
    chunks = chunking_service.chunk_document(doc_id="doc_101", content=content)
    assert len(chunks) >= 1
    assert chunks[0].parent_doc_id == "doc_101"
    assert "Auth Service" in chunks[0].text
