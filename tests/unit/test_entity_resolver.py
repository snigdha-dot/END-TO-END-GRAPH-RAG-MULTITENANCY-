"""Unit tests for entity disambiguation and canonical merging."""
from __future__ import annotations

from app.core.tenant_schema import AI_TRENDS_SCHEMA, MOVIES_SCHEMA
from app.models.graph import Edge, Vertex
from app.services.resolution_service import EntityResolutionService, resolution_service


def _vertex(vid: str, name: str, label: str = "Person") -> Vertex:
    from app.services.extraction_service import normalize_entity_name

    return Vertex(
        id=vid,
        label=label,
        properties={"name": name, "normalized_name": normalize_entity_name(name)},
    )


def test_surface_variants_merge_into_one_canonical_node():
    vertices = [
        _vertex("v1", "Leonardo DiCaprio"),
        _vertex("v2", "leonardo dicaprio"),
        _vertex("v3", "Christopher Nolan"),
    ]
    resolved, _ = resolution_service.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA)
    assert len(resolved) == 2


def test_edges_are_remapped_onto_canonical_ids():
    v1 = _vertex("v1", "Christopher Nolan")
    v2 = _vertex("v2", "christopher nolan")
    v3 = _vertex("v3", "Inception", label="Film")

    edges = [
        Edge(source="v1", target="v3", type="DIRECTED", properties={"confidence": 0.9}),
        Edge(source="v2", target="v3", type="DIRECTED", properties={"confidence": 0.9}),
    ]
    resolved_v, resolved_e = resolution_service.resolve_and_merge(
        [v1, v2, v3], edges, schema=MOVIES_SCHEMA
    )
    assert len(resolved_v) == 2
    # Both edges collapse onto the same canonical pair.
    assert len(resolved_e) == 1
    assert resolved_e[0].source == resolution_service.canonical_id_for(
        "christopher_nolan", "Person"
    )


def test_different_labels_never_merge():
    """The film 'Dune' and the studio 'Dune' are different entities."""
    vertices = [_vertex("v1", "Dune", label="Film"), _vertex("v2", "Dune", label="Studio")]
    resolved, _ = resolution_service.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA)
    assert len(resolved) == 2


def test_distinct_entities_stay_separate():
    vertices = [
        _vertex("v1", "Christopher Nolan"),
        _vertex("v2", "Denis Villeneuve"),
        _vertex("v3", "Steven Spielberg"),
    ]
    resolved, _ = resolution_service.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA)
    assert len(resolved) == 3


def test_aliases_are_recorded_on_the_canonical_node():
    vertices = [_vertex("v1", "Christopher Nolan"), _vertex("v2", "christopher nolan")]
    resolved, _ = resolution_service.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA)
    assert resolved[0].properties.get("mention_count", 0) >= 2


def test_self_loops_are_dropped():
    v1 = _vertex("v1", "GPT-4", label="Model")
    v2 = _vertex("v2", "gpt 4", label="Model")
    edges = [Edge(source="v1", target="v2", type="BUILDS_ON", properties={"confidence": 0.9})]
    _, resolved_e = resolution_service.resolve_and_merge([v1, v2], edges, schema=AI_TRENDS_SCHEMA)
    # Both endpoints resolve to the same canonical node, so the edge is a self-loop.
    assert resolved_e == []


def test_non_schema_labels_are_dropped():
    vertices = [_vertex("v1", "Something", label="NotAMovieLabel")]
    resolved, _ = resolution_service.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA)
    assert resolved == []


def test_duplicate_edges_accumulate_support():
    v1 = _vertex("v1", "Nolan")
    v2 = _vertex("v2", "Inception", label="Film")
    edges = [
        Edge(source="v1", target="v2", type="DIRECTED", properties={"confidence": 0.85}),
        Edge(source="v1", target="v2", type="DIRECTED", properties={"confidence": 0.95}),
    ]
    _, resolved_e = resolution_service.resolve_and_merge([v1, v2], edges, schema=MOVIES_SCHEMA)
    assert len(resolved_e) == 1
    assert resolved_e[0].properties.get("support_count", 1) >= 2


def test_empty_input_is_handled():
    assert resolution_service.resolve_and_merge([], []) == ([], [])


def test_mention_linking_prefers_exact_match():
    candidates = [
        {"entity_id": "canon_inception", "name": "Inception", "label": "Film",
         "normalized_name": "inception", "aliases": []},
        {"entity_id": "canon_interstellar", "name": "Interstellar", "label": "Film",
         "normalized_name": "interstellar", "aliases": []},
    ]
    match = resolution_service.link_mention_to_candidates("Inception", candidates)
    assert match is not None
    assert match["entity_id"] == "canon_inception"
    assert match["method"] == "exact"


def test_mention_linking_returns_none_without_candidates():
    assert resolution_service.link_mention_to_candidates("Inception", []) is None


def test_threshold_is_configurable():
    strict = EntityResolutionService(similarity_threshold=0.99)
    vertices = [_vertex("v1", "Christopher Nolan"), _vertex("v2", "Christopher Nolen")]
    resolved, _ = strict.resolve_and_merge(vertices, [], schema=MOVIES_SCHEMA, use_embeddings=False)
    assert len(resolved) == 2
