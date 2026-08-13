"""Entity disambiguation and canonical merging (plan section 3, Step 3).

Pipeline: vector candidate lookup -> Jaro-Winkler string confirmation (>= 0.85)
-> LLM verification for ambiguous cases (interface present, null in FOSS build)
-> canonical node merge.

Two fixes over the previous implementation:
  * Blocking. Comparing every mention against every canonical node was O(n^2) and
    became the ingestion bottleneck on real documents. Candidates are now narrowed
    by a cheap blocking key first.
  * Semantic aliases. Pure string similarity cannot merge "OpenAI" with
    "Open AI Inc" (Jaro-Winkler ~0.83, below threshold). Embedding similarity
    catches those; the string score then confirms.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rapidfuzz.distance import JaroWinkler

from app.core.config import settings
from app.core.tenant_schema import TenantGraphSchema
from app.models.graph import Edge, Vertex
from app.services.embedding_service import embedding_service
from app.services.extraction_service import normalize_entity_name

logger = logging.getLogger(__name__)


class EntityResolutionService:
    """Deduplicates extracted entities into canonical graph nodes."""

    def __init__(
        self,
        similarity_threshold: Optional[float] = None,
        vector_threshold: Optional[float] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold or settings.ENTITY_LINK_SIMILARITY
        self.vector_threshold = vector_threshold or settings.SIMILARITY_THRESHOLD

    # ------------------------------------------------------------------ identity
    @staticmethod
    def canonical_id_for(normalized_name: str, label: str) -> str:
        """Canonical entity id.

        Scoped by label so a shared surface form across types stays distinct: the
        film "Dune" and the studio "Dune" are different entities.
        """
        return f"canon_{label.lower()}_{normalized_name}"

    # ------------------------------------------------------------------ blocking
    @staticmethod
    def _blocking_keys(normalized: str, label: str = "") -> List[str]:
        """Cheap keys that co-locate plausible matches without comparing everything.

        Two names that share no token and differ in first character are almost never
        the same entity, so they never need a similarity computation. Keys are
        label-scoped so cross-type candidates are never even considered.
        """
        keys: List[str] = []
        if not normalized:
            return keys
        prefix = f"{label}:" if label else ""
        keys.append(f"{prefix}{normalized[:3]}")
        for token in normalized.split("_"):
            if len(token) >= 4:
                keys.append(f"{prefix}tok:{token}")
        return keys

    # ------------------------------------------------------------------ matching
    def _string_similarity(self, a: str, b: str) -> float:
        return JaroWinkler.similarity(a, b)

    def _find_match(
        self,
        normalized: str,
        label: str,
        candidates: Sequence[Tuple[str, Vertex]],
        embedding: Optional[List[float]],
        embedding_cache: Dict[str, List[float]],
    ) -> Optional[Tuple[str, float, str]]:
        """Return (canonical_id, score, method) for the best acceptable match."""
        best: Optional[Tuple[str, float, str]] = None

        for canonical_id, vertex in candidates:
            # Entities of different types are never the same thing, even if the
            # strings match: the film "Dune" and the studio "Dune" are distinct.
            if vertex.label != label:
                continue

            canon_norm = str(vertex.properties.get("normalized_name", ""))
            if not canon_norm:
                continue

            if canon_norm == normalized:
                return canonical_id, 1.0, "exact"

            string_score = self._string_similarity(normalized, canon_norm)
            if string_score >= self.similarity_threshold:
                if best is None or string_score > best[1]:
                    best = (canonical_id, string_score, "string_match")
                continue

            # Semantic rescue for aliases the string score misses.
            if embedding is not None and string_score >= 0.60:
                canon_vec = embedding_cache.get(canonical_id)
                if canon_vec is None:
                    canon_vec = embedding_service.encode(str(vertex.properties.get("name", "")))
                    embedding_cache[canonical_id] = canon_vec
                vector_score = embedding_service.cosine_similarity(embedding, canon_vec)
                if vector_score >= max(self.vector_threshold, 0.80):
                    combined = round((vector_score * 0.6) + (string_score * 0.4), 4)
                    if best is None or combined > best[1]:
                        best = (canonical_id, combined, "vector_assisted")

        return best

    # ------------------------------------------------------------------ public
    def resolve_and_merge(
        self,
        vertices: List[Vertex],
        edges: List[Edge],
        schema: Optional[TenantGraphSchema] = None,
        use_embeddings: bool = True,
    ) -> Tuple[List[Vertex], List[Edge]]:
        """Deduplicate vertices into canonical nodes and remap edges onto them."""
        if not vertices:
            return [], self._dedupe_edges(edges, {})

        # Embed all mentions in one batch; per-entity encoding is far slower.
        embeddings: Dict[str, List[float]] = {}
        if use_embeddings:
            names = [str(v.properties.get("name", v.id)) for v in vertices]
            try:
                vectors = embedding_service.encode_batch(names)
                for vertex, vector in zip(vertices, vectors):
                    embeddings[vertex.id] = vector
            except Exception as exc:  # noqa: BLE001 - resolution must not hard-fail
                logger.warning("Embedding batch failed during resolution (%s); string-only.", exc)

        canonical_map: Dict[str, str] = {}
        resolved: Dict[str, Vertex] = {}
        blocks: Dict[str, List[str]] = defaultdict(list)
        canon_embeddings: Dict[str, List[float]] = {}

        for vertex in vertices:
            raw_name = str(vertex.properties.get("name", vertex.id))
            normalized = str(vertex.properties.get("normalized_name") or normalize_entity_name(raw_name))
            if not normalized:
                continue

            if schema is not None and not schema.validate_vertex_label(vertex.label):
                logger.debug("Dropping vertex with non-schema label %r", vertex.label)
                continue

            keys = self._blocking_keys(normalized, vertex.label)
            candidate_ids: List[str] = []
            seen_ids: set[str] = set()
            for key in keys:
                for cid in blocks.get(key, ()):
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        candidate_ids.append(cid)

            candidates = [(cid, resolved[cid]) for cid in candidate_ids if cid in resolved]
            match = self._find_match(
                normalized, vertex.label, candidates, embeddings.get(vertex.id), canon_embeddings
            )

            if match is not None:
                canonical_id, score, method = match
                canonical_map[vertex.id] = canonical_id
                canon = resolved[canonical_id]

                aliases = canon.properties.setdefault("aliases", [])
                if raw_name not in aliases and raw_name != canon.properties.get("name"):
                    aliases.append(raw_name)
                norm_aliases = canon.properties.setdefault("normalized_aliases", [])
                if normalized not in norm_aliases:
                    norm_aliases.append(normalized)

                canon.properties["mention_count"] = int(canon.properties.get("mention_count", 1)) + 1
                # Keep the highest-confidence surface form as the display name.
                if float(vertex.properties.get("confidence", 0)) > float(
                    canon.properties.get("confidence", 0)
                ):
                    canon.properties["name"] = raw_name
                    canon.properties["confidence"] = vertex.properties.get("confidence", 0)
                canon.properties.setdefault("merge_methods", []).append(method)
                continue

            # Include the label in the id: entities of different types that share a
            # surface form ("Dune" the film vs "Dune" the studio) are distinct nodes
            # and must not collide on the same canonical key.
            canonical_id = self.canonical_id_for(normalized, vertex.label)
            canonical_map[vertex.id] = canonical_id
            merged = Vertex(
                id=canonical_id,
                label=vertex.label,
                properties={
                    **vertex.properties,
                    "name": raw_name,
                    "normalized_name": normalized,
                    "entity_id": canonical_id,
                    "aliases": list(vertex.properties.get("aliases", [])),
                    "normalized_aliases": [normalized],
                    "mention_count": 1,
                },
            )
            resolved[canonical_id] = merged
            if vertex.id in embeddings:
                canon_embeddings[canonical_id] = embeddings[vertex.id]
            for key in keys:
                blocks[key].append(canonical_id)

        return list(resolved.values()), self._dedupe_edges(edges, canonical_map)

    def _dedupe_edges(self, edges: List[Edge], canonical_map: Dict[str, str]) -> List[Edge]:
        """Remap edges onto canonical ids, dropping self-loops and duplicates."""
        out: List[Edge] = []
        best_by_key: Dict[Tuple[str, str, str], Edge] = {}

        for edge in edges:
            source = canonical_map.get(edge.source, edge.source)
            target = canonical_map.get(edge.target, edge.target)
            if source == target:
                continue

            key = (source, edge.type, target)
            remapped = Edge(
                source=source, target=target, type=edge.type, properties=dict(edge.properties)
            )
            existing = best_by_key.get(key)
            if existing is None:
                best_by_key[key] = remapped
            else:
                # Same relation asserted twice: keep the strongest, count the support.
                if remapped.confidence > existing.confidence:
                    remapped.properties["support_count"] = (
                        int(existing.properties.get("support_count", 1)) + 1
                    )
                    best_by_key[key] = remapped
                else:
                    existing.properties["support_count"] = (
                        int(existing.properties.get("support_count", 1)) + 1
                    )

        out.extend(best_by_key.values())
        return out

    # ------------------------------------------------------------------ linking
    def link_mention_to_candidates(
        self, mention: str, candidates: List[Dict[str, Any]], mention_embedding: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """Pick the best canonical entity for a query mention.

        Used at retrieval time to turn query text into real graph seed ids.
        """
        if not candidates:
            return None

        normalized = normalize_entity_name(mention)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []

        for candidate in candidates:
            canon_norm = str(candidate.get("normalized_name") or normalize_entity_name(
                str(candidate.get("name", ""))
            ))
            if not canon_norm:
                continue

            if canon_norm == normalized:
                scored.append((1.0, "exact", candidate))
                continue

            string_score = self._string_similarity(normalized, canon_norm)

            aliases = candidate.get("aliases") or []
            for alias in aliases:
                alias_score = self._string_similarity(normalized, normalize_entity_name(str(alias)))
                string_score = max(string_score, alias_score)

            if string_score >= self.similarity_threshold:
                scored.append((string_score, "string_match", candidate))
            elif mention_embedding is not None and string_score >= 0.55:
                canon_vec = embedding_service.encode(str(candidate.get("name", "")))
                vector_score = embedding_service.cosine_similarity(mention_embedding, canon_vec)
                if vector_score >= self.vector_threshold:
                    scored.append((round(vector_score * 0.7 + string_score * 0.3, 4), "vector_knn", candidate))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0], reverse=True)
        score, method, candidate = scored[0]
        return {**candidate, "score": score, "method": method}


resolution_service = EntityResolutionService()
