"""Hybrid Entity & Relationship Extraction Service."""
import re
from typing import List, Tuple
from app.models.graph import Vertex, Edge, Triple

class ExtractionService:
    def __init__(self):
        # Explicit relationship patterns
        self.patterns = [
            (r'(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)\s+depends on\s+(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)', "DEPENDS_ON"),
            (r'(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)\s+owns\s+(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)', "OWNS"),
            (r'(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)\s+manages\s+(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)', "MANAGES"),
            (r'(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)\s+uses\s+(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)', "DEPENDS_ON"),
            (r'(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)\s+cites\s+(\b[A-Z][a-zA-Z0-9_\s]{2,30}\b)', "CITES"),
        ]

    def extract_from_chunk(self, text: str, chunk_id: str) -> Tuple[List[Vertex], List[Edge]]:
        """Extract vertices and edges from text chunk."""
        vertices: List[Vertex] = []
        edges: List[Edge] = []
        seen_vertices = set()

        # 1. Extract pattern-based triples
        for pattern, rel_type in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                src_name = match.group(1).strip()
                tgt_name = match.group(2).strip()

                src_id = f"ent_{src_name.lower().replace(' ', '_')}"
                tgt_id = f"ent_{tgt_name.lower().replace(' ', '_')}"

                if src_id not in seen_vertices:
                    vertices.append(Vertex(id=src_id, label="Concept", properties={"name": src_name, "provenance": chunk_id}))
                    seen_vertices.add(src_id)

                if tgt_id not in seen_vertices:
                    vertices.append(Vertex(id=tgt_id, label="Concept", properties={"name": tgt_name, "provenance": chunk_id}))
                    seen_vertices.add(tgt_id)

                edges.append(
                    Edge(
                        source=src_id,
                        target=tgt_id,
                        type=rel_type,
                        properties={"confidence": 0.90, "chunk_id": chunk_id}
                    )
                )

        # 2. Extract capitalized entity mentions
        capitalized = re.findall(r'\b[A-Z][a-zA-Z0-9_]{2,20}(?:\s+[A-Z][a-zA-Z0-9_]{2,20})*\b', text)
        for mention in capitalized:
            clean_name = mention.strip()
            ent_id = f"ent_{clean_name.lower().replace(' ', '_')}"
            if ent_id not in seen_vertices and len(clean_name) > 3:
                vertices.append(Vertex(id=ent_id, label="Entity", properties={"name": clean_name, "provenance": chunk_id}))
                seen_vertices.add(ent_id)

        return vertices, edges


extraction_service = ExtractionService()
