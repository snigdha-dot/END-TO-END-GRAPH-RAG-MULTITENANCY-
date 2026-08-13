"""Unit tests for entity linking, RRF fusion, reranking, and the telemetry contract."""
from __future__ import annotations

import pytest

from app.core.telemetry import CONTRACT_STEPS, TelemetryTracker
from app.models.graph import Edge, RetrievedChunk, Subgraph, Vertex
from app.services.reranker_service import RerankerService, reranker_service
from app.services.retrieval_service import HybridRetrievalService


# --------------------------------------------------------------- entity linking
@pytest.fixture
def service() -> HybridRetrievalService:
    return HybridRetrievalService()


def test_stopwords_are_not_treated_as_entity_mentions(service):
    """The original defect: seeds were `canon_which`, `canon_films`, `canon_did`.

    Those match nothing, so the graph path could never return a result.
    """
    mentions = service._candidate_mentions("Which other films did the director of Inception make?")
    lowered = [m.lower() for m in mentions]
    for stopword in ("which", "did", "the", "other", "make"):
        assert stopword not in lowered


def test_real_entity_is_extracted_from_a_multi_hop_query(service):
    mentions = service._candidate_mentions("Which other films did the director of Inception make?")
    assert any("inception" in m.lower() for m in mentions)


def test_multiword_proper_nouns_are_kept_whole(service):
    mentions = service._candidate_mentions("What did Christopher Nolan direct after Interstellar?")
    assert any("christopher nolan" in m.lower() for m in mentions)


def test_lowercase_query_still_yields_mentions(service):
    """Fallback path: users do not always capitalize."""
    mentions = service._candidate_mentions("what is a diffusion model")
    assert mentions
    assert any("diffusion" in m.lower() for m in mentions)


def test_ai_trends_query_extracts_model_names(service):
    mentions = service._candidate_mentions("How does GPT-4 compare to Claude on reasoning?")
    joined = " ".join(mentions).lower()
    assert "gpt" in joined or "claude" in joined


def test_mentions_are_bounded(service):
    long_query = " ".join(f"Entity{i}" for i in range(50))
    assert len(service._candidate_mentions(long_query)) <= 8


# --------------------------------------------------------------- RRF
def test_rrf_ranks_items_found_by_both_paths_highest():
    """The core property that makes dual-path retrieval work."""
    fused = RerankerService.reciprocal_rank_fusion(
        {"vector": ["a", "b", "c"], "graph": ["c", "a", "d"]}
    )
    ranking = [doc_id for doc_id, _ in fused]
    # "a" is rank 1 and 2; "c" is rank 3 and 1. Both beat single-path items.
    assert set(ranking[:2]) == {"a", "c"}


def test_rrf_handles_an_empty_path():
    fused = RerankerService.reciprocal_rank_fusion({"vector": ["a", "b"], "graph": []})
    assert [d for d, _ in fused] == ["a", "b"]


def test_rrf_scores_decrease_with_rank():
    fused = RerankerService.reciprocal_rank_fusion({"vector": ["a", "b", "c"]})
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_weights_are_applied():
    unweighted = dict(RerankerService.reciprocal_rank_fusion({"vector": ["a"], "graph": ["b"]}))
    weighted = dict(
        RerankerService.reciprocal_rank_fusion(
            {"vector": ["a"], "graph": ["b"]}, weights={"graph": 3.0}
        )
    )
    assert weighted["b"] > unweighted["b"]


# --------------------------------------------------------------- reranking
def test_reranking_orders_by_query_relevance():
    chunks = [
        RetrievedChunk(chunk_id="c1", text="Unrelated content about cooking recipes.",
                       parent_doc_id="d"),
        RetrievedChunk(chunk_id="c2",
                       text="Christopher Nolan directed Inception in 2010.", parent_doc_id="d"),
    ]
    ranked = reranker_service.rerank_chunks("Who directed Inception?", chunks, top_k=2)
    assert ranked[0].chunk_id == "c2"


def test_reranking_respects_top_k():
    chunks = [
        RetrievedChunk(chunk_id=f"c{i}", text=f"Content {i}", parent_doc_id="d") for i in range(10)
    ]
    assert len(reranker_service.rerank_chunks("query", chunks, top_k=3)) == 3


def test_reranking_empty_input_is_safe():
    assert reranker_service.rerank_chunks("query", [], top_k=5) == []


def test_fusion_marks_chunks_found_by_both_paths():
    vector = [RetrievedChunk(chunk_id="shared", text="Inception was directed by Nolan.",
                             parent_doc_id="d", retrieval_path="vector")]
    graph = [RetrievedChunk(chunk_id="shared", text="Inception was directed by Nolan.",
                            parent_doc_id="d", retrieval_path="graph")]
    fused = reranker_service.fuse_paths("Who directed Inception?", vector, graph, top_k=5)
    assert fused[0].retrieval_path == "fused"


# --------------------------------------------------------------- pruning
def test_centrality_pruning_keeps_seeds_and_hubs():
    nodes = [Vertex(id=f"n{i}", label="Person", properties={"name": f"N{i}"}) for i in range(10)]
    edges = [Edge(source="n0", target=f"n{i}", type="DIRECTED") for i in range(1, 6)]
    pruned = RerankerService.prune_subgraph_by_centrality(
        Subgraph(nodes=nodes, edges=edges), seed_ids=["n9"], max_nodes=4
    )
    kept = {n.id for n in pruned.nodes}
    assert "n9" in kept  # the seed always survives
    assert "n0" in kept  # the hub survives on degree
    assert len(pruned.nodes) == 4


def test_pruning_is_a_noop_below_the_limit():
    subgraph = Subgraph(nodes=[Vertex(id="n1", label="Person", properties={})], edges=[])
    assert RerankerService.prune_subgraph_by_centrality(subgraph, [], 10) is subgraph


def test_pruning_drops_dangling_edges():
    nodes = [Vertex(id=f"n{i}", label="Person", properties={}) for i in range(5)]
    edges = [Edge(source="n0", target="n4", type="DIRECTED")]
    pruned = RerankerService.prune_subgraph_by_centrality(
        Subgraph(nodes=nodes, edges=edges), seed_ids=["n0"], max_nodes=2
    )
    kept = {n.id for n in pruned.nodes}
    for edge in pruned.edges:
        assert edge.source in kept and edge.target in kept


# --------------------------------------------------------------- telemetry
def test_telemetry_always_emits_the_contract_steps():
    """Plan section 5 declares these keys; Team A codes against them.

    They must be present even when a stage is skipped, or the client contract
    varies by execution path.
    """
    telemetry = TelemetryTracker().finalize()
    for step in CONTRACT_STEPS:
        assert step in telemetry["latency_breakdown_ms"]
    assert "total_retrieval_latency" in telemetry["latency_breakdown_ms"]


def test_arcadedb_vector_knn_key_is_present():
    """Specifically checked: this key was missing from the original response."""
    assert "arcadedb_vector_knn" in TelemetryTracker().finalize()["latency_breakdown_ms"]


def test_foss_models_report_zero_cost():
    tracker = TelemetryTracker()
    tracker.record_model_call("embed", "bge-small-en-v1.5", 1000, 0, 12.5)
    result = tracker.finalize()
    assert result["model_cost_breakdown"]["total_request_cost_usd"] == 0.0
    assert result["model_cost_breakdown"]["models_called"][0]["priced"] is True


def test_priced_model_cost_is_computed():
    tracker = TelemetryTracker()
    tracker.record_model_call("extract", "gemini-1.5-flash", 10_000, 1_000, 200.0)
    total = tracker.finalize()["model_cost_breakdown"]["total_request_cost_usd"]
    assert total > 0


def test_unknown_model_is_flagged_not_guessed():
    tracker = TelemetryTracker()
    tracker.record_model_call("x", "some-unlisted-model", 100, 10, 5.0)
    call = tracker.finalize()["model_cost_breakdown"]["models_called"][0]
    assert call["cost_usd"] == 0.0
    assert call["priced"] is False


def test_time_step_records_even_when_the_block_raises():
    tracker = TelemetryTracker()
    with pytest.raises(ValueError):
        with tracker.time_step("failing_step"):
            raise ValueError("boom")
    assert "failing_step" in tracker.latency_breakdown_ms
