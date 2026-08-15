"""Routes each chunk to the extraction strategy its kind deserves.

    RECORD   Fields are read directly. The column header states the type and the
             cell states the value, so relations are *read* at confidence 1.0
             rather than inferred. Sending a record to an LLM pays to re-derive
             what the source already states.

    TABLE    Each row is treated as a record against the table's headers.

    PROSE    LLM extraction when a provider is configured, heuristic extraction
             otherwise. Prose is where inference is genuinely required, and where
             an LLM is worth its cost.

Everything converges on the same schema validation, so output is uniform
regardless of which path produced it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.core.config import settings
from app.core.tenant_schema import TenantGraphSchema
from app.models.canonical import CanonicalChunk, ChunkKind
from app.models.graph import Edge, Vertex
from app.services.extraction_service import (
    entity_id_for,
    extraction_service,
    normalize_entity_name,
)
from app.services.llm_extraction import (
    LLMExtractionProvider,
    get_llm_provider,
    schema_validator,
)

logger = logging.getLogger(__name__)

_MULTI_VALUE_SPLIT = re.compile(r"\s*[;,/|]\s*|\s+and\s+", re.IGNORECASE)
_NULL_VALUES = {
    "", "na", "n/a", "none", "null", "nil", "-", "--", "unknown",
    "not applicable", "not specified", "varies", "nan",
}

# Column-name families mapped to the generic edge they imply.
_COLUMN_EDGE_HINTS: List[Tuple[str, str]] = [
    (r"\b(?:symptom|effect|complication|risk|cause|treat|remedy|herb|medicine|"
     r"drug|intervention|therapy|prevention|indication)\b", "AFFECTS"),
    (r"\b(?:ingredient|component|part|contains|composition|includes)\b", "PART_OF"),
    (r"\b(?:source|origin|derived|extracted|obtained|from)\b", "DERIVED_FROM"),
    (r"\b(?:reference|citation|cited|bibliography)\b", "CITES"),
    (r"\b(?:history|prior|previous|before|after|next|stage|progression)\b", "PRECEDES"),
]

# Column-name families whose values are categorical rather than entities.
_ATTRIBUTE_HINT = re.compile(
    r"\b(?:severity|duration|frequency|type|category|class|group|status|level|"
    r"stage|gender|sex|season|seasonal|dosha|doshas|prakriti|constitution|age|"
    r"grade|rating|score|region|pattern|patterns|allerg)\b",
    re.IGNORECASE,
)

# Columns that translate another column's value rather than naming a new entity.
_ALIAS_HINT = re.compile(
    r"\b(?:hindi|marathi|tamil|telugu|bengali|sanskrit|urdu|arabic|chinese|"
    r"spanish|french|german|latin|local|native|vernacular|regional|alternate)\s+"
    r"(?:name|title|term)\b|\b(?:name|title)\s+in\s+\w+\b",
    re.IGNORECASE,
)

# Columns holding prose rather than a name.
_FREETEXT_HINT = re.compile(
    r"\b(?:description|summary|notes?|comment|text|content|detail|recommendation|"
    r"recommendations|instruction|advice|prognosis|diagnosis|guideline)\b",
    re.IGNORECASE,
)


@dataclass
class ExtractionResult:
    """Entities and relations from one chunk, plus how they were obtained."""

    entities: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    method: str = "none"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    coerced: int = 0
    rejected: int = 0


def _is_null(value: Any) -> bool:
    return value is None or str(value).strip().lower() in _NULL_VALUES


def _split_values(raw: str) -> List[str]:
    if not raw:
        return []
    parts = [p.strip(" .;,") for p in _MULTI_VALUE_SPLIT.split(raw)]
    return [p for p in parts if p and p.lower() not in _NULL_VALUES]


class RecordExtractor:
    """Derives entities and relations from a record chunk's fields."""

    MAX_VALUES_PER_FIELD = 8

    def extract(
        self, chunk: CanonicalChunk, schema: TenantGraphSchema
    ) -> ExtractionResult:
        fields = chunk.fields
        if not fields:
            return ExtractionResult(method="record")

        columns = list(fields.keys())
        subject_column = self._pick_subject(columns, chunk)
        result = ExtractionResult(method="record")

        vertices: Dict[str, Vertex] = {}

        def add(value: str, label: str) -> Optional[str]:
            normalized = normalize_entity_name(value)
            if not normalized or not schema.validate_vertex_label(label):
                return None
            entity_id = entity_id_for(value, label)
            if entity_id not in vertices:
                vertices[entity_id] = Vertex(
                    id=entity_id,
                    label=label,
                    properties={
                        "name": value,
                        "normalized_name": normalized,
                        "entity_id": entity_id,
                        # Read from a column, not inferred from prose.
                        "confidence": 1.0,
                        "provenance": chunk.chunk_id,
                        "extractor": "record",
                    },
                )
            return entity_id

        subject_ids: List[str] = []
        if subject_column and not _is_null(fields.get(subject_column)):
            label = self._entity_label(subject_column, schema)
            for value in _split_values(str(fields[subject_column]))[: self.MAX_VALUES_PER_FIELD]:
                entity_id = add(value, label)
                if entity_id:
                    subject_ids.append(entity_id)

        # Translated names become aliases of the subject, not new nodes: promoting
        # them creates one duplicate per language for the same real entity.
        aliases: List[str] = []
        for column, raw in fields.items():
            if _ALIAS_HINT.search(column) and not _is_null(raw):
                aliases.extend(_split_values(str(raw)))
        if aliases:
            for subject_id in subject_ids:
                existing = vertices[subject_id].properties.setdefault("aliases", [])
                for alias in aliases:
                    if alias not in existing:
                        existing.append(alias)

        for column, raw in fields.items():
            if column == subject_column or _is_null(raw) or _ALIAS_HINT.search(column):
                continue
            # Prose columns are not entities; they stay in the chunk text where
            # vector search can still reach them.
            if _FREETEXT_HINT.search(column) or len(str(raw)) > 120:
                continue

            is_attribute = bool(_ATTRIBUTE_HINT.search(column))
            label = "Attribute" if is_attribute else self._entity_label(column, schema)
            edge_type = (
                "HAS_ATTRIBUTE" if is_attribute else self._edge_for_column(column, schema)
            )
            if not schema.validate_edge_type(edge_type):
                continue

            for value in _split_values(str(raw))[: self.MAX_VALUES_PER_FIELD]:
                entity_id = add(value, label)
                if not entity_id:
                    continue
                for subject_id in subject_ids:
                    if subject_id == entity_id:
                        continue
                    result.edges.append(
                        Edge(
                            source=subject_id,
                            target=entity_id,
                            type=edge_type,
                            properties={
                                "confidence": 1.0,
                                "source_column": column,
                                "chunk_id": chunk.chunk_id,
                                "extractor": "record",
                            },
                        )
                    )

        result.entities = list(vertices.values())
        return result

    @staticmethod
    def _pick_subject(columns: Sequence[str], chunk: CanonicalChunk) -> Optional[str]:
        """Choose the column the row is *about*.

        This anchors every relation in the row, so choosing wrong inverts the
        graph. The adapter records the subject when it knows it; otherwise the
        first non-alias, non-attribute column wins, since tabular data
        conventionally leads with its subject.
        """
        declared = chunk.metadata.get("subject")
        if declared:
            for column in columns:
                if str(chunk.fields.get(column, "")).strip() == str(declared).strip():
                    return column

        for column in columns:
            if _ALIAS_HINT.search(column) or _ATTRIBUTE_HINT.search(column):
                continue
            if _FREETEXT_HINT.search(column):
                continue
            return column
        return columns[0] if columns else None

    @staticmethod
    def _entity_label(column: str, schema: TenantGraphSchema) -> str:
        lowered = column.lower()
        candidates: List[Tuple[str, str]] = [
            (r"\b(?:person|author|doctor|patient|practitioner|name)\b", "Person"),
            (r"\b(?:company|organization|institution|manufacturer|brand|publisher)\b",
             "Organization"),
            (r"\b(?:product|formulation|remedy|medicine|drug|item|document|book)\b",
             "Artifact"),
            (r"\b(?:disease|condition|symptom|concept|topic|technique|method|"
             r"therapy|treatment|practice|factor)\b", "Concept"),
        ]
        for pattern, label in candidates:
            if re.search(pattern, lowered) and label in schema.vertex_labels:
                return label
        return "Entity" if "Entity" in schema.vertex_labels else next(iter(schema.vertex_labels))

    @staticmethod
    def _edge_for_column(column: str, schema: TenantGraphSchema) -> str:
        lowered = column.lower()
        for pattern, edge_type in _COLUMN_EDGE_HINTS:
            if re.search(pattern, lowered) and schema.validate_edge_type(edge_type):
                return edge_type
        return (
            "ASSOCIATED_WITH"
            if schema.validate_edge_type("ASSOCIATED_WITH")
            else "RELATED_TO"
        )


class TableExtractor:
    """Treats each table row as a record against the table's headers."""

    MAX_ROWS = 200

    def extract(
        self, chunk: CanonicalChunk, schema: TenantGraphSchema
    ) -> ExtractionResult:
        headers = chunk.table_headers
        if not headers:
            return ExtractionResult(method="table")

        rows = self._parse_rows(chunk.text, len(headers))
        record_extractor = RecordExtractor()
        merged = ExtractionResult(method="table")
        seen_entities: Set[str] = set()
        seen_edges: Set[Tuple[str, str, str]] = set()

        for index, values in enumerate(rows[: self.MAX_ROWS]):
            row_chunk = CanonicalChunk(
                chunk_id=f"{chunk.chunk_id}_r{index}",
                doc_id=chunk.doc_id,
                kind=ChunkKind.RECORD,
                text="",
                token_count=0,
                provenance=chunk.provenance,
                fields=dict(zip(headers, values)),
                table_headers=headers,
            )
            result = record_extractor.extract(row_chunk, schema)
            for vertex in result.entities:
                if vertex.id not in seen_entities:
                    seen_entities.add(vertex.id)
                    merged.entities.append(vertex)
            for edge in result.edges:
                key = (edge.source, edge.type, edge.target)
                if key not in seen_edges:
                    seen_edges.add(key)
                    merged.edges.append(edge)

        return merged

    @staticmethod
    def _parse_rows(text: str, column_count: int) -> List[List[str]]:
        rows: List[List[str]] = []
        for line in text.splitlines():
            if "|" not in line or re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == column_count:
                rows.append(cells)
        # The first matching row is the header itself.
        return rows[1:] if rows else []


class ExtractionRouter:
    """Sends each chunk to the strategy its kind deserves."""

    def __init__(self, provider: Optional[LLMExtractionProvider] = None) -> None:
        self.record_extractor = RecordExtractor()
        self.table_extractor = TableExtractor()
        self._provider = provider or get_llm_provider()

    @property
    def llm_available(self) -> bool:
        return self._provider.is_available

    @property
    def prose_method(self) -> str:
        return "llm" if self.llm_available else extraction_service.active_backend

    async def extract(
        self, chunk: CanonicalChunk, schema: TenantGraphSchema
    ) -> ExtractionResult:
        if chunk.kind is ChunkKind.RECORD:
            return self.record_extractor.extract(chunk, schema)
        if chunk.kind is ChunkKind.TABLE:
            return self.table_extractor.extract(chunk, schema)
        return await self._extract_prose(chunk, schema)

    async def _extract_prose(
        self, chunk: CanonicalChunk, schema: TenantGraphSchema
    ) -> ExtractionResult:
        if self.llm_available:
            proposal = await self._provider.propose(chunk.text, schema)
            outcome = schema_validator.validate(proposal, schema, chunk.chunk_id)
            return ExtractionResult(
                entities=outcome.entities,
                edges=outcome.edges,
                method="llm",
                prompt_tokens=proposal.prompt_tokens,
                completion_tokens=proposal.completion_tokens,
                coerced=outcome.coerced,
                rejected=outcome.rejected,
            )

        vertices, edges = extraction_service.extract_from_chunk(
            chunk.text, chunk.chunk_id, schema
        )
        return ExtractionResult(
            entities=vertices, edges=edges, method=extraction_service.active_backend
        )

    async def extract_many(
        self, chunks: Sequence[CanonicalChunk], schema: TenantGraphSchema
    ) -> Tuple[List[Vertex], List[Edge], List[Tuple[str, str]], Dict[str, Any]]:
        """Extract across chunks, returning entities, edges, mentions, and stats."""
        all_vertices: List[Vertex] = []
        all_edges: List[Edge] = []
        mentions: List[Tuple[str, str]] = []
        stats = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "coerced": 0, "rejected": 0, "methods": {},
        }

        for chunk in chunks:
            result = await self.extract(chunk, schema)
            all_vertices.extend(result.entities)
            all_edges.extend(result.edges)
            for vertex in result.entities:
                mentions.append((vertex.id, chunk.chunk_id))

            stats["prompt_tokens"] += result.prompt_tokens
            stats["completion_tokens"] += result.completion_tokens
            stats["coerced"] += result.coerced
            stats["rejected"] += result.rejected
            stats["methods"][result.method] = stats["methods"].get(result.method, 0) + 1

        return all_vertices, all_edges, mentions, stats


extraction_router = ExtractionRouter()
