"""Unit tests for schema-driven entity and relation extraction."""
from __future__ import annotations

from app.core.tenant_schema import AI_TRENDS_SCHEMA, MOVIES_SCHEMA
from app.services.extraction_service import (
    entity_id_for,
    extraction_service,
    normalize_entity_name,
)

MOVIE_TEXT = (
    "Inception was directed by Christopher Nolan. "
    "Leonardo DiCaprio starred in Inception. "
    "Warner Bros produced the film."
)

AI_TEXT = (
    "GPT-4 was released by OpenAI. "
    "GPT-4 builds on the Transformer architecture. "
    "The model was trained on WebText."
)


def test_movie_entities_use_the_movie_schema():
    vertices, _ = extraction_service.extract_from_chunk(MOVIE_TEXT, "c1", MOVIES_SCHEMA)
    assert vertices
    for vertex in vertices:
        assert MOVIES_SCHEMA.validate_vertex_label(vertex.label)


def test_ai_entities_use_the_ai_schema():
    vertices, _ = extraction_service.extract_from_chunk(AI_TEXT, "c1", AI_TRENDS_SCHEMA)
    assert vertices
    for vertex in vertices:
        assert AI_TRENDS_SCHEMA.validate_vertex_label(vertex.label)


def test_edges_never_leave_the_tenant_vocabulary():
    """A movies edge type must never appear in an AI-trends extraction."""
    _, movie_edges = extraction_service.extract_from_chunk(MOVIE_TEXT, "c1", MOVIES_SCHEMA)
    for edge in movie_edges:
        assert MOVIES_SCHEMA.validate_edge_type(edge.type)

    _, ai_edges = extraction_service.extract_from_chunk(AI_TEXT, "c1", AI_TRENDS_SCHEMA)
    for edge in ai_edges:
        assert AI_TRENDS_SCHEMA.validate_edge_type(edge.type)
        assert edge.type != "DIRECTED"


def test_low_confidence_edges_are_filtered():
    from app.core.config import settings

    _, edges = extraction_service.extract_from_chunk(MOVIE_TEXT, "c1", MOVIES_SCHEMA)
    for edge in edges:
        assert edge.confidence >= settings.EDGE_CONFIDENCE_THRESHOLD


def test_edges_carry_provenance():
    _, edges = extraction_service.extract_from_chunk(MOVIE_TEXT, "chunk_42", MOVIES_SCHEMA)
    for edge in edges:
        assert edge.properties.get("chunk_id") == "chunk_42"


def test_empty_text_yields_nothing():
    assert extraction_service.extract_from_chunk("", "c1", MOVIES_SCHEMA) == ([], [])
    assert extraction_service.extract_from_chunk("   ", "c1", MOVIES_SCHEMA) == ([], [])


def test_no_self_loop_edges():
    vertices, edges = extraction_service.extract_from_chunk(
        "Inception depends on Inception.", "c1", MOVIES_SCHEMA
    )
    for edge in edges:
        assert edge.source != edge.target


def test_normalization_is_stable():
    assert normalize_entity_name("Christopher Nolan") == "christopher_nolan"
    assert normalize_entity_name("Warner Bros.") == "warner_bros"
    # Hyphens and spaces normalize identically, so surface variants converge.
    assert normalize_entity_name("  GPT-4!  ") == normalize_entity_name("GPT 4")


def test_entity_ids_are_label_scoped():
    """Same surface form, different type: must not collide."""
    assert entity_id_for("Dune", "Film") != entity_id_for("Dune", "Studio")


def test_section_header_noise_is_not_extracted():
    """'Plot' and 'Reception' are headings, not entities."""
    vertices, _ = extraction_service.extract_from_chunk(
        "Plot\n\nReception\n\nOverview", "c1", MOVIES_SCHEMA
    )
    names = {v.properties["name"].lower() for v in vertices}
    assert "plot" not in names
    assert "reception" not in names


def test_extraction_reports_its_backend():
    assert extraction_service.active_backend in ("gliner", "spacy", "regex")
    assert extraction_service.model_label
