"""Community detection and reports (the GraphRAG global-search layer).

Communities answer a different question from entity traversal. Traversal answers
"what treats cough?" by walking from a named seed. Communities answer "what are
the main themes in this knowledge base?" — questions with no entity to anchor on,
where the answer is a summary of a region of the graph rather than a path.

Two stages:

  detect   Leiden when `graspologic` is installed, label propagation otherwise.
           Both are free and deterministic given a seed; neither needs an LLM.

  report   An LLM writes a readable summary per community when a provider is
           configured. Without one, an extractive report is assembled from the
           community's own entities and relations — less fluent, but factual and
           free, and it keeps the retrieval path working in the FOSS-only build.

Reports are stored as Chunk vertices so global search reuses the existing vector
index rather than needing a second retrieval mechanism.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.tenant_schema import TenantGraphSchema
from app.models.graph import Edge, Vertex
from app.services.llm_extraction import LLMExtractionProvider, get_llm_provider

logger = logging.getLogger(__name__)

REPORT_PROMPT = """Summarize this community of related entities as a short report.

Entities:
{entities}

Relationships:
{relations}

Write:
1. A title of at most 8 words naming what this community is about.
2. A 2-4 sentence summary of the theme connecting these entities.

Return ONLY valid JSON: {{"title": "...", "summary": "..."}}
"""


@dataclass
class Community:
    """A cluster of densely connected entities."""

    community_id: str
    level: int
    entity_ids: List[str] = field(default_factory=list)
    edge_keys: List[Tuple[str, str, str]] = field(default_factory=list)
    title: str = ""
    summary: str = ""
    rank: float = 0.0

    @property
    def size(self) -> int:
        return len(self.entity_ids)

    def report_text(self) -> str:
        return f"# {self.title}\n\n{self.summary}" if self.title else self.summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "size": self.size,
            "edges": len(self.edge_keys),
            "title": self.title,
            "rank": round(self.rank, 4),
        }


class CommunityDetector:
    """Partitions a graph into communities."""

    MIN_COMMUNITY_SIZE = 3
    MAX_COMMUNITIES = 200

    def detect(
        self, entities: Sequence[Vertex], edges: Sequence[Edge], level: int = 0
    ) -> List[Community]:
        """Partition entities into communities using the best available algorithm."""
        if len(entities) < self.MIN_COMMUNITY_SIZE or not edges:
            return []

        adjacency = self._build_adjacency(entities, edges)
        partition = self._leiden(adjacency) or self._label_propagation(adjacency)

        grouped: Dict[Any, List[str]] = defaultdict(list)
        for entity_id, cluster in partition.items():
            grouped[cluster].append(entity_id)

        edges_by_member: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
        for edge in edges:
            key = (edge.source, edge.type, edge.target)
            edges_by_member[edge.source].append(key)

        communities: List[Community] = []
        for index, (_, members) in enumerate(
            sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
        ):
            if len(members) < self.MIN_COMMUNITY_SIZE:
                continue
            if len(communities) >= self.MAX_COMMUNITIES:
                break

            member_set = set(members)
            internal = [
                key
                for member in members
                for key in edges_by_member.get(member, [])
                if key[2] in member_set
            ]
            communities.append(
                Community(
                    community_id=f"community_L{level}_{index}",
                    level=level,
                    entity_ids=sorted(members),
                    edge_keys=internal,
                    # Rank by internal density: a community whose members are
                    # heavily interconnected is a real theme, not a coincidence.
                    rank=len(internal) / max(1, len(members)),
                )
            )

        return communities

    @staticmethod
    def _build_adjacency(
        entities: Sequence[Vertex], edges: Sequence[Edge]
    ) -> Dict[str, Set[str]]:
        known = {v.id for v in entities}
        adjacency: Dict[str, Set[str]] = {v.id: set() for v in entities}
        for edge in edges:
            if edge.source in known and edge.target in known and edge.source != edge.target:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        return adjacency

    def _leiden(self, adjacency: Dict[str, Set[str]]) -> Optional[Dict[str, int]]:
        """Leiden partitioning via graspologic, when installed."""
        try:
            import networkx as nx  # noqa: PLC0415
            from graspologic.partition import hierarchical_leiden  # noqa: PLC0415
        except ImportError:
            return None

        try:
            graph = nx.Graph()
            for node, neighbours in adjacency.items():
                graph.add_node(node)
                for neighbour in neighbours:
                    graph.add_edge(node, neighbour)

            if graph.number_of_edges() == 0:
                return None

            partitions = hierarchical_leiden(graph, max_cluster_size=25, random_seed=42)
            return {p.node: p.cluster for p in partitions if p.is_final_cluster}
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail
            logger.warning("Leiden partitioning failed (%s); using label propagation.", exc)
            return None

    def _label_propagation(self, adjacency: Dict[str, Set[str]]) -> Dict[str, int]:
        """Deterministic label propagation.

        Each node adopts the most common label among its neighbours, ties broken by
        the smallest label so runs are reproducible. Not as good as Leiden at
        finding modular structure, but dependency-free and adequate for grouping.
        """
        labels: Dict[str, int] = {
            node: index for index, node in enumerate(sorted(adjacency))
        }

        for _ in range(12):
            changed = False
            for node in sorted(adjacency):
                neighbours = adjacency[node]
                if not neighbours:
                    continue
                counts: Dict[int, int] = defaultdict(int)
                for neighbour in neighbours:
                    counts[labels[neighbour]] += 1
                best = min(
                    counts.items(), key=lambda kv: (-kv[1], kv[0])
                )[0]
                if labels[node] != best:
                    labels[node] = best
                    changed = True
            if not changed:
                break

        return labels


class CommunityReporter:
    """Produces a title and summary for each community."""

    MAX_ENTITIES_IN_PROMPT = 30
    MAX_RELATIONS_IN_PROMPT = 40

    def __init__(self, provider: Optional[LLMExtractionProvider] = None) -> None:
        self._provider = provider or get_llm_provider()

    @property
    def llm_available(self) -> bool:
        return self._provider.is_available

    async def write_reports(
        self,
        communities: Sequence[Community],
        entities_by_id: Dict[str, Vertex],
        edges_by_key: Dict[Tuple[str, str, str], Edge],
    ) -> List[Community]:
        for community in communities:
            names = [
                entities_by_id[e].properties.get("name", e)
                for e in community.entity_ids
                if e in entities_by_id
            ]
            relations = [
                self._describe_edge(key, entities_by_id)
                for key in community.edge_keys[: self.MAX_RELATIONS_IN_PROMPT]
            ]
            relations = [r for r in relations if r]

            title, summary = "", ""
            if self.llm_available:
                title, summary = await self._llm_report(names, relations)

            # An empty LLM result (transport error, unparseable JSON) falls through
            # to the extractive report rather than leaving a community unlabelled.
            if not summary:
                title, summary = self._extractive_report(
                    names, relations, entities_by_id, community
                )

            community.title = title
            community.summary = summary

        return list(communities)

    @staticmethod
    def _describe_edge(
        key: Tuple[str, str, str], entities_by_id: Dict[str, Vertex]
    ) -> str:
        source_id, edge_type, target_id = key
        source = entities_by_id.get(source_id)
        target = entities_by_id.get(target_id)
        if not source or not target:
            return ""
        relation = edge_type.replace("_", " ").lower()
        return f"{source.properties.get('name', source_id)} {relation} {target.properties.get('name', target_id)}"

    async def _llm_report(
        self, names: Sequence[str], relations: Sequence[str]
    ) -> Tuple[str, str]:
        prompt = REPORT_PROMPT.format(
            entities=", ".join(names[: self.MAX_ENTITIES_IN_PROMPT]),
            relations="\n".join(relations),
        )
        parsed = await self._provider.complete_json(prompt)
        return str(parsed.get("title", ""))[:80], str(parsed.get("summary", ""))

    @staticmethod
    def _extractive_report(
        names: Sequence[str],
        relations: Sequence[str],
        entities_by_id: Dict[str, Vertex],
        community: Community,
    ) -> Tuple[str, str]:
        """Assemble a factual report without an LLM.

        Less fluent than a generated summary, but it states only what the graph
        contains, and it keeps global search working in the FOSS-only build.
        """
        if not names:
            return "", ""

        # The most-connected entities name the community better than the largest.
        degree: Dict[str, int] = defaultdict(int)
        for source_id, _, target_id in community.edge_keys:
            degree[source_id] += 1
            degree[target_id] += 1
        central = sorted(
            community.entity_ids, key=lambda e: degree.get(e, 0), reverse=True
        )[:3]
        central_names = [
            entities_by_id[e].properties.get("name", e)
            for e in central
            if e in entities_by_id
        ]

        title = " · ".join(central_names[:2]) if central_names else names[0]

        label_counts: Dict[str, int] = defaultdict(int)
        for entity_id in community.entity_ids:
            vertex = entities_by_id.get(entity_id)
            if vertex:
                label_counts[vertex.label] += 1
        composition = ", ".join(
            f"{count} {label}" for label, count in sorted(
                label_counts.items(), key=lambda kv: kv[1], reverse=True
            )
        )

        summary_parts = [
            f"A group of {len(names)} related entities ({composition}) "
            f"centred on {', '.join(central_names) or names[0]}."
        ]
        if relations:
            summary_parts.append("Key relationships: " + "; ".join(relations[:6]) + ".")
        summary_parts.append("Members include: " + ", ".join(names[:15]) + ".")

        return title[:80], " ".join(summary_parts)


class CommunityService:
    """Detects communities and produces their reports."""

    def __init__(self) -> None:
        self.detector = CommunityDetector()
        self.reporter = CommunityReporter()

    async def build(
        self,
        entities: Sequence[Vertex],
        edges: Sequence[Edge],
        schema: TenantGraphSchema,
        level: int = 0,
    ) -> List[Community]:
        """Detect communities and attach a report to each."""
        communities = self.detector.detect(entities, edges, level=level)
        if not communities:
            return []

        entities_by_id = {v.id: v for v in entities}
        edges_by_key = {(e.source, e.type, e.target): e for e in edges}
        communities = await self.reporter.write_reports(
            communities, entities_by_id, edges_by_key
        )

        logger.info(
            "Built %d communities (largest %d entities, LLM reports: %s)",
            len(communities),
            max((c.size for c in communities), default=0),
            self.reporter.llm_available,
        )
        return communities

    @staticmethod
    def to_chunk_statements(
        communities: Sequence[Community], vectors: Sequence[Sequence[float]]
    ) -> List[Dict[str, Any]]:
        """Store reports as Chunk vertices.

        Reusing the chunk type means global search runs through the existing
        vector index instead of needing a second retrieval mechanism.
        """
        statements: List[Dict[str, Any]] = []
        for community, vector in zip(communities, vectors):
            statements.append(
                {
                    "command": (
                        "MERGE (c:Chunk {chunk_id: $chunk_id}) "
                        "SET c.text = $text, c.parent_doc_id = $parent_doc_id, "
                        "c.chunk_kind = 'community_report', c.embedding = $embedding, "
                        "c.community_level = $level, c.community_size = $size, "
                        "c.community_rank = $rank, c.citation = $citation"
                    ),
                    "params": {
                        "chunk_id": community.community_id,
                        "text": community.report_text(),
                        "parent_doc_id": "communities",
                        "embedding": list(vector),
                        "level": community.level,
                        "size": community.size,
                        "rank": community.rank,
                        "citation": f"Community report · {community.title}",
                    },
                }
            )
        return statements


community_service = CommunityService()
