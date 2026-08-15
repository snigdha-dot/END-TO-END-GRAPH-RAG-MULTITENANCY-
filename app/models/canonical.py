"""Canonical document model: the convergence point for every source format.

Every adapter — structured, prose, rich — produces this one shape, so the rest of
the pipeline has a single input contract. Format handling stops at the adapter
boundary rather than leaking into chunking, extraction, and writing.

The previous design had two parallel pipelines (`ingest_document` for prose,
`ingest_structured` for tables) that duplicated resolution, validation, and the
write path. Anything fixed in one had to be fixed in the other.

A document is a sequence of *blocks*, each carrying its own kind. A block knows
whether it is prose, a table, a record, or a heading, which lets the chunker treat
each appropriately instead of flattening everything to text and losing the
structure that made it retrievable.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BlockKind(str, Enum):
    """What a block of content actually is.

    This distinction survives all the way to chunking: a table split like prose
    loses its header row, and a record split at all separates a fact from the
    subject it describes.
    """

    PROSE = "prose"
    HEADING = "heading"
    TABLE = "table"
    RECORD = "record"      # one row of a structured source
    LIST = "list"
    CODE = "code"
    CAPTION = "caption"


class Provenance(BaseModel):
    """Where a piece of content came from.

    Attached at parse time, not after validation: when a chunk is rejected you
    need to know which page of which file produced it, which means the metadata
    had to exist before the rejection.
    """

    source_uri: str = Field(description="File path or URL of the origin")
    source_format: str = Field(description="csv, pdf, md, docx, html, ...")
    page: Optional[int] = Field(default=None, description="1-indexed page, for paged formats")
    row_index: Optional[int] = Field(default=None, description="0-indexed row, for tabular sources")
    section_path: List[str] = Field(default_factory=list, description="Enclosing heading breadcrumb")
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    extractor: str = Field(default="", description="Which adapter produced this")

    def describe(self) -> str:
        """A short human-readable citation, for passages returned to a caller."""
        parts = [self.source_uri.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.row_index is not None:
            parts.append(f"row {self.row_index}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return " · ".join(parts)


class TableCell(BaseModel):
    """One cell, retaining its column so a table survives serialization."""

    column: str
    value: str


class ContentBlock(BaseModel):
    """One structurally coherent unit of a document."""

    block_id: str
    kind: BlockKind
    text: str = Field(description="Plain-text rendering; always populated")
    provenance: Provenance

    # Populated for HEADING blocks.
    heading_level: Optional[int] = None

    # Populated for TABLE and RECORD blocks. Keeping the fields separate from the
    # rendered text is what lets the record path derive entities from columns
    # instead of re-extracting them from prose.
    fields: List[TableCell] = Field(default_factory=list)
    table_headers: List[str] = Field(default_factory=list)
    table_rows: List[List[str]] = Field(default_factory=list)

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_structured(self) -> bool:
        return self.kind in (BlockKind.TABLE, BlockKind.RECORD)

    def field_map(self) -> Dict[str, str]:
        """Fields as a column -> value mapping."""
        return {c.column: c.value for c in self.fields}


class CanonicalDocument(BaseModel):
    """One source document, normalized. The single input to the chunker."""

    doc_id: str
    title: str = ""
    blocks: List[ContentBlock] = Field(default_factory=list)
    provenance: Provenance
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(b.text.strip() for b in self.blocks)

    @property
    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    def blocks_of(self, *kinds: BlockKind) -> List[ContentBlock]:
        return [b for b in self.blocks if b.kind in kinds]

    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for block in self.blocks:
            counts[block.kind.value] = counts.get(block.kind.value, 0) + 1
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "blocks": len(self.blocks),
            "block_kinds": counts,
            "chars": self.total_chars,
            "source": self.provenance.source_uri,
            "format": self.provenance.source_format,
        }


class ChunkKind(str, Enum):
    """How a chunk should be embedded, extracted from, and displayed."""

    PROSE = "prose"
    TABLE = "table"
    RECORD = "record"


class ValidationIssue(BaseModel):
    """Why a chunk was rejected or flagged."""

    code: str
    detail: str
    severity: str = "reject"  # reject | warn


class CanonicalChunk(BaseModel):
    """A retrievable unit, typed by what it contains.

    Carries its provenance so a returned passage can be cited, and its source
    fields so a record chunk yields entities from columns rather than from NER.
    """

    chunk_id: str
    doc_id: str
    kind: ChunkKind
    text: str
    token_count: int
    provenance: Provenance

    section_path: List[str] = Field(default_factory=list)
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    has_overlap: bool = False

    # Structured payload, preserved for TABLE and RECORD chunks.
    fields: Dict[str, str] = Field(default_factory=dict)
    table_headers: List[str] = Field(default_factory=list)

    issues: List[ValidationIssue] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "reject" for i in self.issues)

    def embedding_text(self) -> str:
        """Text as it should be embedded.

        Prose gets its section breadcrumb prepended: a chunk reading "It grossed
        $836M" is unanswerable alone but answerable under "Inception > Reception".
        Structured chunks are already self-describing, since each value is
        rendered beside its column name.
        """
        if self.kind is ChunkKind.PROSE and self.section_path:
            return " > ".join(self.section_path) + "\n\n" + self.text
        return self.text
