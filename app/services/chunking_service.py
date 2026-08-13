"""Structure-aware semantic chunking with overlap and parent-child hierarchy.

Plan section 3, Step 1: markdown/paragraph structure awareness, 400-600 token chunks,
100-token overlap, and child chunks linked to parent document context.

Three things the previous version got wrong:
  * no overlap at all, so a fact spanning a boundary was lost from both chunks;
  * `len(text) // 4` token estimation, which drifts badly on real prose;
  * `parent_doc_id` as a bare string, with no section hierarchy to give an LLM context.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

# Markdown ATX headings, captured with level so hierarchy can be reconstructed.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)
_WORD_RE = re.compile(r"\w+|[^\w\s]")


class DocumentChunk(BaseModel):
    """One retrievable unit of text plus the context needed to interpret it."""

    chunk_id: str
    parent_doc_id: str
    chunk_index: int
    text: str
    token_count: int
    # Breadcrumb of enclosing markdown headings, e.g. ["Inception", "Production"].
    section_path: List[str] = Field(default_factory=list)
    # Immediately preceding/following chunk ids: parent-child sibling linkage.
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    # True when this chunk's leading tokens are duplicated from the previous chunk.
    has_overlap: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def contextual_text(self) -> str:
        """Text prefixed with its section breadcrumb, for embedding and for the LLM.

        A chunk reading "It grossed $836M" is useless in isolation; prefixed with
        "Inception > Reception" it is answerable.
        """
        if not self.section_path:
            return self.text
        return " > ".join(self.section_path) + "\n\n" + self.text


class _Tokenizer:
    """Token counting with the real tokenizer when available, words otherwise."""

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
        # ~1.3 tokens per word is a far better estimate for English than chars/4.
        words = len(_WORD_RE.findall(text))
        return max(1, int(words * 1.3)) if words else 0

    def split_tail(self, text: str, token_budget: int) -> str:
        """Return the trailing ~`token_budget` tokens of `text`, on a word boundary."""
        self._load()
        if token_budget <= 0 or not text:
            return ""
        if self._encoder is not None:
            try:
                ids = self._encoder.encode(text)
                if len(ids) <= token_budget:
                    return text
                return self._encoder.decode(ids[-token_budget:])
            except Exception:  # noqa: BLE001
                pass
        words = text.split()
        approx_words = max(1, int(token_budget / 1.3))
        return " ".join(words[-approx_words:])


_tokenizer = _Tokenizer()


class ChunkingService:
    """Splits documents into overlapping, hierarchy-aware semantic chunks."""

    def __init__(
        self,
        target_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> None:
        self.target_tokens = target_tokens or settings.CHUNK_TARGET_TOKENS
        self.max_tokens = max_tokens or settings.CHUNK_MAX_TOKENS
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP_TOKENS

    # ------------------------------------------------------------------ helpers
    def _segment(self, content: str) -> List[Dict[str, Any]]:
        """Split into paragraph blocks, each tagged with its heading breadcrumb."""
        segments: List[Dict[str, Any]] = []
        heading_stack: List[tuple[int, str]] = []
        position = 0

        for match in _HEADING_RE.finditer(content):
            body = content[position : match.start()].strip()
            if body:
                segments.extend(self._paragraphs(body, [h[1] for h in heading_stack]))

            level = len(match.group(1))
            title = match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            position = match.end()

        trailing = content[position:].strip()
        if trailing:
            segments.extend(self._paragraphs(trailing, [h[1] for h in heading_stack]))

        if not segments:
            stripped = content.strip()
            if stripped:
                segments = self._paragraphs(stripped, [])
        return segments

    def _paragraphs(self, block: str, section_path: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for para in re.split(r"\n\s*\n", block):
            text = para.strip()
            if not text:
                continue
            out.append(
                {
                    "text": text,
                    "section_path": list(section_path),
                    "tokens": _tokenizer.count(text),
                }
            )
        return out

    def _split_oversized(self, segment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break a single paragraph larger than max_tokens on sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", segment["text"])
        parts: List[Dict[str, Any]] = []
        buffer: List[str] = []
        buffer_tokens = 0

        for sentence in sentences:
            s_tokens = _tokenizer.count(sentence)
            if buffer and buffer_tokens + s_tokens > self.target_tokens:
                parts.append(
                    {
                        "text": " ".join(buffer),
                        "section_path": list(segment["section_path"]),
                        "tokens": buffer_tokens,
                    }
                )
                buffer, buffer_tokens = [], 0
            buffer.append(sentence)
            buffer_tokens += s_tokens

        if buffer:
            parts.append(
                {
                    "text": " ".join(buffer),
                    "section_path": list(segment["section_path"]),
                    "tokens": buffer_tokens,
                }
            )
        return parts

    # ------------------------------------------------------------------ chunking
    def chunk_document(
        self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Split a document into overlapping, section-aware chunks."""
        meta = dict(metadata or {})
        if not content or not content.strip():
            return []

        segments: List[Dict[str, Any]] = []
        for seg in self._segment(content):
            if seg["tokens"] > self.max_tokens:
                segments.extend(self._split_oversized(seg))
            else:
                segments.append(seg)

        # Pack segments into chunks, never crossing a section boundary.
        packed: List[Dict[str, Any]] = []
        buffer: List[Dict[str, Any]] = []
        buffer_tokens = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens
            if buffer:
                packed.append(
                    {
                        "text": "\n\n".join(s["text"] for s in buffer),
                        "section_path": list(buffer[0]["section_path"]),
                        "tokens": buffer_tokens,
                    }
                )
                buffer, buffer_tokens = [], 0

        for seg in segments:
            section_changed = bool(buffer) and seg["section_path"] != buffer[0]["section_path"]
            would_overflow = buffer_tokens + seg["tokens"] > self.target_tokens
            if buffer and (section_changed or would_overflow):
                flush()
            buffer.append(seg)
            buffer_tokens += seg["tokens"]
        flush()

        # Apply overlap and wire sibling links.
        chunks: List[DocumentChunk] = []
        for idx, item in enumerate(packed):
            text = item["text"]
            has_overlap = False
            if idx > 0 and self.overlap_tokens > 0:
                tail = _tokenizer.split_tail(packed[idx - 1]["text"], self.overlap_tokens)
                if tail:
                    text = f"{tail}\n\n{text}"
                    has_overlap = True

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    parent_doc_id=doc_id,
                    chunk_index=idx,
                    text=text,
                    token_count=_tokenizer.count(text),
                    section_path=item["section_path"],
                    prev_chunk_id=f"{doc_id}_chunk_{idx - 1}" if idx > 0 else None,
                    next_chunk_id=f"{doc_id}_chunk_{idx + 1}" if idx < len(packed) - 1 else None,
                    has_overlap=has_overlap,
                    metadata=meta,
                )
            )

        return chunks


chunking_service = ChunkingService()
