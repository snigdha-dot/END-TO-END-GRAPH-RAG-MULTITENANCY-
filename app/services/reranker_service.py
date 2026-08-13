"""Reciprocal Rank Fusion and cross-encoder reranking (plan section 4, Stage 3).

RRF fuses the two retrieval paths without needing their scores to be comparable —
vector cosine similarity and graph hop-distance live on different scales, so fusing
by *rank* rather than score is what makes a dual-path system work at all.

    RRF(d) = sum over paths of  1 / (k + rank_path(d))

The cross-encoder then rescores the fused top slice by reading each candidate
against the query jointly, which is more accurate than bi-encoder similarity and
too slow to run on anything but a short list.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.models.graph import RetrievedChunk, Subgraph, Vertex

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


class RerankerService:
    """Fuses ranked lists and optionally rescores with a cross-encoder."""

    def __init__(self) -> None:
        self._cross_encoder = None
        self._load_attempted = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    def _load_cross_encoder(self) -> None:
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            if not settings.RERANKER_ENABLED:
                return
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                logger.info("Loading cross-encoder '%s'...", settings.CROSS_ENCODER_MODEL)
                self._cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Cross-encoder unavailable (%s); using lexical overlap rescoring.", exc
                )

    @property
    def has_cross_encoder(self) -> bool:
        self._load_cross_encoder()
        return self._cross_encoder is not None

    @property
    def model_label(self) -> str:
        return settings.CROSS_ENCODER_LABEL if self.has_cross_encoder else "lexical-fallback"

    # ------------------------------------------------------------------ RRF
    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: Dict[str, List[str]], k: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, float]]:
        """Fuse named ranked id-lists into one ranking.

        `ranked_lists` maps a path name ("vector", "graph") to ids in rank order.
        Returns (id, fused_score) sorted best-first.
        """
        k = k or settings.RRF_K
        weights = weights or {}
        scores: Dict[str, float] = defaultdict(float)

        for path, ids in ranked_lists.items():
            weight = weights.get(path, 1.0)
            for rank, doc_id in enumerate(ids, start=1):
                scores[doc_id] += weight * (1.0 / (k + rank))

        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    # ------------------------------------------------------------------ scoring
    @staticmethod
    def _lexical_score(query: str, text: str) -> float:
        """Overlap-based relevance used when the cross-encoder is unavailable."""
        q_terms = set(_WORD_RE.findall(query.lower()))
        d_terms = _WORD_RE.findall(text.lower())
        if not q_terms or not d_terms:
            return 0.0
        d_set = set(d_terms)
        overlap = len(q_terms & d_set)
        coverage = overlap / len(q_terms)
        # Mild length normalization so long chunks do not win on volume alone.
        density = overlap / math.sqrt(len(d_terms)) if d_terms else 0.0
        return round((coverage * 0.75) + min(density, 1.0) * 0.25, 4)

    def rerank_chunks(
        self, query: str, chunks: List[RetrievedChunk], top_k: int
    ) -> List[RetrievedChunk]:
        """Rescore candidates against the query and return the best `top_k`."""
        if not chunks:
            return []

        self._load_cross_encoder()
        # Bound cross-encoder work: it is O(n) model calls.
        candidates = chunks[: max(top_k * 4, 20)]

        if self._cross_encoder is not None:
            try:
                pairs = [(query, c.text) for c in candidates]
                raw = self._cross_encoder.predict(pairs)
                for chunk, score in zip(candidates, raw):
                    # Squash logits into (0,1) so scores are comparable across queries.
                    chunk.score = round(1.0 / (1.0 + math.exp(-float(score))), 4)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Cross-encoder inference failed (%s); lexical rescoring.", exc)
                for chunk in candidates:
                    chunk.score = self._lexical_score(query, chunk.text)
        else:
            for chunk in candidates:
                chunk.score = self._lexical_score(query, chunk.text)

        candidates.sort(key=lambda c: c.score, reverse=True)
        for rank, chunk in enumerate(candidates, start=1):
            chunk.rank = rank
        return candidates[:top_k]

    # ------------------------------------------------------------------ pruning
    @staticmethod
    def prune_subgraph_by_centrality(
        subgraph: Subgraph, seed_ids: Sequence[str], max_nodes: int
    ) -> Subgraph:
        """Keep the most connected nodes, always retaining the query's seeds.

        Plan Stage 3: "Filter irrelevant nodes based on centrality". A 2-hop
        traversal from a well-connected seed can return far more than an LLM can
        use; degree centrality keeps the structurally important part.
        """
        if len(subgraph.nodes) <= max_nodes:
            return subgraph

        degree: Dict[str, int] = defaultdict(int)
        for edge in subgraph.edges:
            degree[edge.source] += 1
            degree[edge.target] += 1

        seeds = set(seed_ids)
        ranked = sorted(
            subgraph.nodes,
            key=lambda n: (n.id in seeds, degree.get(n.id, 0)),
            reverse=True,
        )
        keep_nodes = ranked[:max_nodes]
        keep_ids = {n.id for n in keep_nodes}
        keep_edges = [
            e for e in subgraph.edges if e.source in keep_ids and e.target in keep_ids
        ]
        return Subgraph(nodes=keep_nodes, edges=keep_edges)

    # ------------------------------------------------------------------ fusion
    def fuse_paths(
        self,
        query: str,
        vector_chunks: List[RetrievedChunk],
        graph_chunks: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Fuse vector and graph candidates via RRF, then rerank the fused list."""
        by_id: Dict[str, RetrievedChunk] = {}
        for chunk in vector_chunks + graph_chunks:
            existing = by_id.get(chunk.chunk_id)
            if existing is None:
                by_id[chunk.chunk_id] = chunk
            elif existing.retrieval_path != chunk.retrieval_path:
                # Surfaced by both paths: strong signal, mark it as fused.
                existing.retrieval_path = "fused"

        ranked_lists = {
            "vector": [c.chunk_id for c in vector_chunks],
            "graph": [c.chunk_id for c in graph_chunks],
        }
        fused = self.reciprocal_rank_fusion(ranked_lists)

        ordered: List[RetrievedChunk] = []
        for chunk_id, rrf_score in fused:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            chunk.rrf_score = round(rrf_score, 6)
            ordered.append(chunk)

        return self.rerank_chunks(query, ordered, top_k)


reranker_service = RerankerService()
