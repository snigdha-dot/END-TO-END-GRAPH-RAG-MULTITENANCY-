"""Context optimizer: packs retrieved material into a caller's token budget.

Returning `top_k` passages ignores that a caller has a finite context window. Ten
long record chunks can exceed 4,000 tokens on their own, and the caller then
truncates arbitrarily — usually dropping whatever came last, which is not
whatever mattered least.

Four stages:

  deduplicate   near-identical chunks waste budget saying the same thing twice.
                Overlapping chunks share their boundary text by design, so exact
                matching is not enough; Jaccard over token sets catches them.
  budget        pack greedily by score until the budget is spent, truncating the
                final passage on a sentence boundary rather than mid-word.
  order         graph relationships first: they carry the multi-hop answer that
                no single chunk states.
  cite          each passage keeps its provenance, so the caller can attribute.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.models.graph import RetrievedChunk, Subgraph

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+|[^\w\s]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class OptimizedContext:
    """What the caller receives, and what it cost."""

    passages: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    chunks: List[RetrievedChunk] = field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    dropped_duplicates: int = 0
    dropped_budget: int = 0
    truncated: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passages": len(self.passages),
            "total_tokens": self.total_tokens,
            "budget_tokens": self.budget_tokens,
            "budget_used_pct": (
                round(100 * self.total_tokens / self.budget_tokens, 1)
                if self.budget_tokens else 0.0
            ),
            "dropped_duplicates": self.dropped_duplicates,
            "dropped_budget": self.dropped_budget,
            "truncated": self.truncated,
        }


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
            pass

    def count(self, text: str) -> int:
        self._load()
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:  # noqa: BLE001
                pass
        words = len(_WORD_RE.findall(text))
        return max(1, int(words * 1.3)) if words else 0

    def truncate(self, text: str, budget: int) -> str:
        """Cut to `budget` tokens, preferring a sentence boundary."""
        if budget <= 0:
            return ""
        if self.count(text) <= budget:
            return text

        self._load()
        if self._encoder is not None:
            try:
                ids = self._encoder.encode(text)
                rough = self._encoder.decode(ids[:budget])
            except Exception:  # noqa: BLE001
                rough = " ".join(text.split()[: int(budget / 1.3)])
        else:
            rough = " ".join(text.split()[: int(budget / 1.3)])

        # Prefer ending on a complete sentence, but not at the cost of most of
        # the passage.
        sentences = _SENTENCE_END.split(rough)
        if len(sentences) > 1:
            trimmed = " ".join(sentences[:-1])
            if len(trimmed) > len(rough) * 0.6:
                return trimmed.rstrip() + " …"
        return rough.rstrip() + " …"


_tokenizer = _Tokenizer()


class ContextOptimizer:
    """Fits retrieved material into a token budget without arbitrary truncation."""

    DEFAULT_BUDGET = 3000
    MIN_PASSAGE_TOKENS = 24
    DUPLICATE_THRESHOLD = 0.82

    def optimize(
        self,
        chunks: Sequence[RetrievedChunk],
        subgraph: Optional[Subgraph] = None,
        seed_ids: Optional[Sequence[str]] = None,
        budget_tokens: Optional[int] = None,
        max_passages: int = 20,
    ) -> OptimizedContext:
        budget = budget_tokens or self.DEFAULT_BUDGET
        context = OptimizedContext(budget_tokens=budget)

        # Graph relationships first: a multi-hop answer lives in the edges, and no
        # single chunk states it. Spending budget here buys what retrieval alone
        # cannot supply.
        graph_passages = self._verbalize_graph(subgraph, seed_ids or [], limit=max_passages // 2)
        for passage in graph_passages:
            tokens = _tokenizer.count(passage)
            if context.total_tokens + tokens > budget:
                context.dropped_budget += 1
                continue
            context.passages.append(passage)
            context.citations.append("graph relationship")
            context.total_tokens += tokens

        deduplicated, duplicates = self._deduplicate(chunks)
        context.dropped_duplicates = duplicates

        for chunk in deduplicated:
            if len(context.passages) >= max_passages:
                context.dropped_budget += 1
                continue

            text = chunk.text.strip()
            if not text:
                continue

            tokens = _tokenizer.count(text)
            remaining = budget - context.total_tokens

            if tokens > remaining:
                # Truncate only if enough would survive to be useful; otherwise
                # drop it and leave the budget for a passage that fits.
                if remaining >= self.MIN_PASSAGE_TOKENS:
                    text = _tokenizer.truncate(text, remaining)
                    tokens = _tokenizer.count(text)
                    context.truncated += 1
                else:
                    context.dropped_budget += 1
                    continue

            context.passages.append(text)
            context.citations.append(self._citation(chunk))
            context.chunks.append(chunk)
            context.total_tokens += tokens

            if context.total_tokens >= budget:
                break

        return context

    # ------------------------------------------------------------------ parts
    def _deduplicate(
        self, chunks: Sequence[RetrievedChunk]
    ) -> Tuple[List[RetrievedChunk], int]:
        """Drop near-identical chunks, keeping the higher-scoring one.

        Chunks overlap by design, so consecutive ones share boundary text and
        exact-match deduplication misses them. Jaccard over token sets catches
        the overlap without discarding genuinely distinct passages.
        """
        kept: List[RetrievedChunk] = []
        kept_tokens: List[Set[str]] = []
        seen_ids: Set[str] = set()
        dropped = 0

        ordered = sorted(chunks, key=lambda c: c.score, reverse=True)

        for chunk in ordered:
            if chunk.chunk_id in seen_ids:
                dropped += 1
                continue
            seen_ids.add(chunk.chunk_id)

            tokens = set(_WORD_RE.findall(chunk.text.lower()))
            if not tokens:
                continue

            is_duplicate = False
            for existing in kept_tokens:
                overlap = len(tokens & existing)
                union = len(tokens | existing)
                if union and overlap / union >= self.DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    break

            if is_duplicate:
                dropped += 1
                continue

            kept.append(chunk)
            kept_tokens.append(tokens)

        return kept, dropped

    @staticmethod
    def _verbalize_graph(
        subgraph: Optional[Subgraph], seed_ids: Sequence[str], limit: int
    ) -> List[str]:
        """Render graph edges as readable statements."""
        if subgraph is None or not subgraph.edges:
            return []

        by_id = {n.id: n for n in subgraph.nodes}
        seeds = set(seed_ids)

        # Edges touching a query seed answer the question most directly.
        ordered = sorted(
            subgraph.edges,
            key=lambda e: (e.source in seeds or e.target in seeds, e.confidence),
            reverse=True,
        )

        passages: List[str] = []
        for edge in ordered[:limit]:
            source = by_id.get(edge.source)
            target = by_id.get(edge.target)
            if source is None or target is None:
                continue
            relation = edge.type.replace("_", " ").lower()
            passages.append(
                f"{source.name} ({source.label}) {relation} {target.name} ({target.label})."
            )
        return passages

    @staticmethod
    def _citation(chunk: RetrievedChunk) -> str:
        parts = [chunk.parent_doc_id or chunk.chunk_id]
        if chunk.section_path:
            parts.append(" > ".join(chunk.section_path))
        parts.append(f"via {chunk.retrieval_path}")
        return " · ".join(p for p in parts if p)


context_optimizer = ContextOptimizer()
