"""Entity Disambiguation and Merging Service."""
from typing import List, Dict, Tuple
from rapidfuzz.distance import JaroWinkler
from app.models.graph import Vertex, Edge

class EntityResolutionService:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def normalize_name(self, name: str) -> str:
        """Normalize raw entity string."""
        return name.strip().lower().replace("-", "_").replace(" ", "_")

    def resolve_and_merge(
        self, vertices: List[Vertex], edges: List[Edge]
    ) -> Tuple[List[Vertex], List[Edge]]:
        """Deduplicate entity nodes and re-map edges to canonical node IDs."""
        canonical_map: Dict[str, str] = {} # raw_id -> canonical_id
        resolved_vertices: Dict[str, Vertex] = {}

        for vertex in vertices:
            raw_id = vertex.id
            raw_name = vertex.properties.get("name", raw_id)
            norm_name = self.normalize_name(raw_name)

            # Check if similar canonical vertex already exists
            matched_canonical_id = None
            for can_id, can_vertex in resolved_vertices.items():
                can_name = can_vertex.properties.get("name", "")
                sim = JaroWinkler.similarity(norm_name, self.normalize_name(can_name))
                if sim >= self.similarity_threshold:
                    matched_canonical_id = can_id
                    break

            if matched_canonical_id:
                canonical_map[raw_id] = matched_canonical_id
                # Merge aliases into existing canonical node
                aliases = resolved_vertices[matched_canonical_id].properties.setdefault("aliases", [])
                if raw_name not in aliases:
                    aliases.append(raw_name)
            else:
                canonical_id = f"canon_{norm_name}"
                canonical_map[raw_id] = canonical_id
                vertex.id = canonical_id
                resolved_vertices[canonical_id] = vertex

        # Re-map edges to canonical node IDs
        resolved_edges: List[Edge] = []
        seen_edges = set()

        for edge in edges:
            new_src = canonical_map.get(edge.source, edge.source)
            new_tgt = canonical_map.get(edge.target, edge.target)

            if new_src != new_tgt: # Avoid self-loops
                edge_key = (new_src, edge.type, new_tgt)
                if edge_key not in seen_edges:
                    edge.source = new_src
                    edge.target = new_tgt
                    seen_edges.add(edge_key)
                    resolved_edges.append(edge)

        return list(resolved_vertices.values()), resolved_edges


resolution_service = EntityResolutionService()
