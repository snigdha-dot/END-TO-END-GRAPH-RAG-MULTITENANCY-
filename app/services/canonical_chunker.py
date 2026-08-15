"""Structure-aware chunker: CanonicalDocument -> typed CanonicalChunks.

Each block kind is chunked according to what it is, because treating them alike
destroys the structure that made them retrievable:

    PROSE    packed to a token target with overlap, never crossing a section
             boundary; oversized paragraphs split on sentence boundaries.
    RECORD   never split. A row is already one semantic unit, and splitting it
             separates a fact from the subject it describes.
    TABLE    split by rows with the header repeated on every part, so a fragment
             is still interpretable. A table chunked like prose loses its header
             after the first piece and the remaining rows become unreadable.

Chunk validation runs after chunking and before embedding, so empty, boilerplate,
and degenerate chunks never reach the index or the graph.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence

from app.core.config import settings
from app.models.canonical import (
    BlockKind,
    CanonicalChunk,
    CanonicalDocument,
    ChunkKind,
    ContentBlock,
    Provenance,
    ValidationIssue,
)

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Content that survives extraction but carries no retrievable signal.
_BOILERPLATE = re.compile(
    r"^(?:see also|references|external links|further reading|notes|bibliography|"
    r"table of contents|contents|index|appendix|page \d+|figure \d+|"
    r"copyright|all rights reserved|confidential)[\s:.-]*$",
    re.IGNORECASE,
)


class _Tokenizer:
    """Token counting via tiktoken when available, words otherwise."""

    def __init__(self) -> None:
        self._encoder = None
        self._attempted = False

    def _load(self) -> None:
        if self._attempted:
            return
        self._attempted = True
        try:
            import tiktoken  # noqa: PLC0415

            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001
            logger.info("tiktoken unavailable; using word-based token estimation.")

    def count(self, text: str) -> int:
        self._load()
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:  # noqa: BLE001
                pass
        # ~1.3 tokens per word is far closer for English than chars/4.
        words = len(_WORD_RE.findall(text))
        return max(1, int(words * 1.3)) if words else 0

    def tail(self, text: str, budget: int) -> str:
        """Trailing ~`budget` tokens, cut on a word boundary."""
        self._load()
        if budget <= 0 or not text:
            return ""
        if self._encoder is not None:
            try:
                ids = self._encoder.encode(text)
                if len(ids) <= budget:
                    return text
                return self._encoder.decode(ids[-budget:])
            except Exception:  # noqa: BLE001
                pass
        words = text.split()
        return " ".join(words[-max(1, int(budget / 1.3)) :])


_tokenizer = _Tokenizer()


class ChunkValidator:
    """Rejects chunks that would add noise to the index or the graph."""

    MIN_TOKENS = 8
    MIN_ALPHA_RATIO = 0.35

    def validate(self, chunk: CanonicalChunk) -> CanonicalChunk:
        issues: List[ValidationIssue] = []
        text = chunk.text.strip()

        if not text:
            issues.append(ValidationIssue(code="empty", detail="Chunk has no content."))
            chunk.issues = issues
            return chunk

        # Record chunks are exempt from the length floor: a short row is still a
        # complete fact, whereas a short prose fragment usually is not.
        if chunk.kind is not ChunkKind.RECORD and chunk.token_count < self.MIN_TOKENS:
            issues.append(
                ValidationIssue(
                    code="too_short",
                    detail=f"{chunk.token_count} tokens is below the {self.MIN_TOKENS} floor.",
                )
            )

        if _BOILERPLATE.match(text):
            issues.append(
                ValidationIssue(code="boilerplate", detail="Matches a navigational heading.")
            )

        alpha = sum(c.isalpha() for c in text)
        if text and alpha / len(text) < self.MIN_ALPHA_RATIO:
            issues.append(
                ValidationIssue(
                    code="low_alpha",
                    detail="Mostly punctuation, digits, or markup.",
                    severity="warn" if chunk.kind is not ChunkKind.PROSE else "reject",
                )
            )

        if chunk.token_count > settings.CHUNK_MAX_TOKENS * 3:
            issues.append(
                ValidationIssue(
                    code="oversized",
                    detail=f"{chunk.token_count} tokens far exceeds the target.",
                    severity="warn",
                )
            )

        chunk.issues = issues
        return chunk


class CanonicalChunker:
    """Turns a CanonicalDocument into validated, typed chunks."""

    def __init__(
        self,
        target_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
        table_rows_per_chunk: int = 20,
    ) -> None:
        self.target_tokens = target_tokens or settings.CHUNK_TARGET_TOKENS
        self.max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
        self.overlap_tokens = (
            overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP_TOKENS
        )
        self.table_rows_per_chunk = table_rows_per_chunk
        self.validator = ChunkValidator()

    # ------------------------------------------------------------------ public
    def chunk(self, document: CanonicalDocument) -> List[CanonicalChunk]:
        """Chunk a document, routing each block kind to its own strategy."""
        chunks: List[CanonicalChunk] = []
        counter = 0

        prose_buffer: List[ContentBlock] = []

        def flush_prose() -> None:
            nonlocal counter, prose_buffer
            if prose_buffer:
                produced = self._chunk_prose(document, prose_buffer, counter)
                counter += len(produced)
                chunks.extend(produced)
                prose_buffer = []

        for block in document.blocks:
            if block.kind is BlockKind.RECORD:
                flush_prose()
                chunks.append(self._chunk_record(document, block, counter))
                counter += 1
            elif block.kind is BlockKind.TABLE:
                flush_prose()
                produced = self._chunk_table(document, block, counter)
                counter += len(produced)
                chunks.extend(produced)
            elif block.kind is BlockKind.HEADING:
                # A heading is context for what follows, not a chunk of its own.
                flush_prose()
            else:
                prose_buffer.append(block)

        flush_prose()

        self._link_siblings(chunks)
        return [self.validator.validate(c) for c in chunks]

    def chunk_valid(self, document: CanonicalDocument) -> List[CanonicalChunk]:
        """Chunk and drop anything that fails validation."""
        chunks = self.chunk(document)
        valid = [c for c in chunks if c.is_valid]
        rejected = len(chunks) - len(valid)
        if rejected:
            logger.info(
                "Chunk validation rejected %d of %d chunks for '%s'.",
                rejected, len(chunks), document.doc_id,
            )
        return valid

    # ------------------------------------------------------------------ record
    def _chunk_record(
        self, document: CanonicalDocument, block: ContentBlock, index: int
    ) -> CanonicalChunk:
        """One record, one chunk. Never split."""
        return CanonicalChunk(
            chunk_id=f"{document.doc_id}_c{index}",
            doc_id=document.doc_id,
            kind=ChunkKind.RECORD,
            text=block.text,
            token_count=_tokenizer.count(block.text),
            provenance=block.provenance,
            fields=block.field_map(),
            table_headers=block.table_headers,
            metadata=block.metadata,
        )

    # ------------------------------------------------------------------- table
    def _chunk_table(
        self, document: CanonicalDocument, block: ContentBlock, start_index: int
    ) -> List[CanonicalChunk]:
        """Split a table by rows, repeating the header on every part.

        Without the repeated header, every chunk after the first is a grid of
        values with no column names — unreadable to both a reader and an LLM.
        """
        headers = block.table_headers
        rows = block.table_rows

        if not rows:
            return [
                CanonicalChunk(
                    chunk_id=f"{document.doc_id}_c{start_index}",
                    doc_id=document.doc_id,
                    kind=ChunkKind.TABLE,
                    text=block.text,
                    token_count=_tokenizer.count(block.text),
                    provenance=block.provenance,
                    table_headers=headers,
                    section_path=block.provenance.section_path,
                )
            ]

        header_line = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join("---" for _ in headers) + " |"

        chunks: List[CanonicalChunk] = []
        for offset in range(0, len(rows), self.table_rows_per_chunk):
            slice_rows = rows[offset : offset + self.table_rows_per_chunk]
            body = "\n".join("| " + " | ".join(r) + " |" for r in slice_rows)
            text = f"{header_line}\n{separator}\n{body}"
            chunks.append(
                CanonicalChunk(
                    chunk_id=f"{document.doc_id}_c{start_index + len(chunks)}",
                    doc_id=document.doc_id,
                    kind=ChunkKind.TABLE,
                    text=text,
                    token_count=_tokenizer.count(text),
                    provenance=block.provenance,
                    table_headers=headers,
                    section_path=block.provenance.section_path,
                    metadata={"row_offset": offset, "row_count": len(slice_rows)},
                )
            )
        return chunks

    # ------------------------------------------------------------------- prose
    def _chunk_prose(
        self, document: CanonicalDocument, blocks: Sequence[ContentBlock], start_index: int
    ) -> List[CanonicalChunk]:
        """Pack prose blocks to the token target, never crossing a section."""
        segments: List[Dict] = []
        for block in blocks:
            tokens = _tokenizer.count(block.text)
            if tokens > self.max_tokens:
                segments.extend(self._split_oversized(block))
            else:
                segments.append(
                    {
                        "text": block.text,
                        "tokens": tokens,
                        "section_path": block.provenance.section_path,
                        "provenance": block.provenance,
                    }
                )

        packed: List[Dict] = []
        buffer: List[Dict] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if buffer:
                packed.append(
                    {
                        "text": "\n\n".join(s["text"] for s in buffer),
                        "section_path": buffer[0]["section_path"],
                        "provenance": buffer[0]["provenance"],
                    }
                )
                buffer, buffer_tokens = [], 0

        for segment in segments:
            section_changed = bool(buffer) and segment["section_path"] != buffer[0]["section_path"]
            would_overflow = buffer_tokens + segment["tokens"] > self.target_tokens
            if buffer and (section_changed or would_overflow):
                flush()
            buffer.append(segment)
            buffer_tokens += segment["tokens"]
        flush()

        chunks: List[CanonicalChunk] = []
        for position, item in enumerate(packed):
            text = item["text"]
            has_overlap = False
            if position > 0 and self.overlap_tokens > 0:
                tail = _tokenizer.tail(packed[position - 1]["text"], self.overlap_tokens)
                if tail:
                    text = f"{tail}\n\n{text}"
                    has_overlap = True

            chunks.append(
                CanonicalChunk(
                    chunk_id=f"{document.doc_id}_c{start_index + position}",
                    doc_id=document.doc_id,
                    kind=ChunkKind.PROSE,
                    text=text,
                    token_count=_tokenizer.count(text),
                    provenance=item["provenance"],
                    section_path=item["section_path"],
                    has_overlap=has_overlap,
                )
            )
        return chunks

    def _split_oversized(self, block: ContentBlock) -> List[Dict]:
        """Break an oversized paragraph on sentence boundaries."""
        sentences = _SENTENCE_SPLIT.split(block.text)
        parts: List[Dict] = []
        buffer: List[str] = []
        buffer_tokens = 0

        for sentence in sentences:
            tokens = _tokenizer.count(sentence)
            if buffer and buffer_tokens + tokens > self.target_tokens:
                parts.append(
                    {
                        "text": " ".join(buffer),
                        "tokens": buffer_tokens,
                        "section_path": block.provenance.section_path,
                        "provenance": block.provenance,
                    }
                )
                buffer, buffer_tokens = [], 0
            buffer.append(sentence)
            buffer_tokens += tokens

        if buffer:
            parts.append(
                {
                    "text": " ".join(buffer),
                    "tokens": buffer_tokens,
                    "section_path": block.provenance.section_path,
                    "provenance": block.provenance,
                }
            )
        return parts

    @staticmethod
    def _link_siblings(chunks: List[CanonicalChunk]) -> None:
        """Wire prev/next so a retrieved chunk can pull its neighbours for context."""
        for position, chunk in enumerate(chunks):
            if position > 0:
                chunk.prev_chunk_id = chunks[position - 1].chunk_id
            if position < len(chunks) - 1:
                chunk.next_chunk_id = chunks[position + 1].chunk_id


canonical_chunker = CanonicalChunker()
