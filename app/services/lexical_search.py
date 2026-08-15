"""BM25 lexical search over tenant chunks.

Dense vectors are weak exactly where lexical matching is strong: exact
identifiers, product codes, drug names, acronyms, and quoted phrases. An
embedding of "GPT-4" and one of "GPT-3" sit close together, which is right for
similarity and wrong for lookup.

BM25 is implemented in-tree rather than pulled from a dependency: the scoring
function is a dozen lines, and the index has to be tenant-scoped and cache-aware
in ways a generic library would not know about.

The index is built per tenant and cached, invalidated by chunk count. A rebuilt
index costs one full scan, so a tenant whose corpus is static pays it once.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.config import settings
from app.models.graph import RetrievedChunk
from app.services.arcadedb_client import arcadedb_client

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")

# Terms so common they contribute nothing to ranking but cost time to score.
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with this these those there their they them then than""".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, keeping hyphenated identifiers intact.

    `gpt-4` must survive as one token: splitting it into `gpt` and `4` is what
    makes lexical search fail on exactly the queries it exists to serve.
    """
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


@dataclass
class _Document:
    chunk_id: str
    text: str
    parent_doc_id: str
    section_path: List[str]
    length: int
    term_frequencies: Dict[str, int]


@dataclass
class BM25Index:
    """An inverted index over one tenant's chunks."""

    tenant_id: str
    documents: List[_Document] = field(default_factory=list)
    document_frequency: Dict[str, int] = field(default_factory=dict)
    average_length: float = 0.0
    built_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self.documents)


class LexicalSearchService:
    """BM25 ranking over tenant chunks, with a per-tenant cached index."""

    K1 = 1.5   # term-frequency saturation
    B = 0.75   # length normalization
    MAX_INDEXED_CHUNKS = 20_000
    CACHE_TTL_SECONDS = 900

    def __init__(self) -> None:
        self._indexes: Dict[str, BM25Index] = {}

    # ------------------------------------------------------------------ index
    async def _load_chunks(self, tenant_id: str) -> List[Dict]:
        rows = await arcadedb_client.execute_sql(
            "SELECT chunk_id, text, parent_doc_id, section_path FROM Chunk LIMIT :limit",
            {"limit": self.MAX_INDEXED_CHUNKS},
            tenant_id=tenant_id,
        )
        return [r for r in rows if isinstance(r, dict) and r.get("chunk_id")]

    async def build_index(self, tenant_id: str, force: bool = False) -> BM25Index:
        """Build or reuse the tenant's index."""
        cached = self._indexes.get(tenant_id)
        if cached and not force and (time.time() - cached.built_at) < self.CACHE_TTL_SECONDS:
            return cached

        started = time.perf_counter()
        rows = await self._load_chunks(tenant_id)

        documents: List[_Document] = []
        document_frequency: Dict[str, int] = defaultdict(int)

        for row in rows:
            text = str(row.get("text", ""))
            tokens = tokenize(text)
            if not tokens:
                continue
            frequencies = Counter(tokens)
            documents.append(
                _Document(
                    chunk_id=str(row["chunk_id"]),
                    text=text,
                    parent_doc_id=str(row.get("parent_doc_id", "")),
                    section_path=list(row.get("section_path") or []),
                    length=len(tokens),
                    term_frequencies=dict(frequencies),
                )
            )
            for term in frequencies:
                document_frequency[term] += 1

        index = BM25Index(
            tenant_id=tenant_id,
            documents=documents,
            document_frequency=dict(document_frequency),
            average_length=(
                sum(d.length for d in documents) / len(documents) if documents else 0.0
            ),
            built_at=time.time(),
        )
        self._indexes[tenant_id] = index

        logger.info(
            "BM25 index built for '%s': %d chunks, %d terms, %.0fms",
            tenant_id, index.size, len(index.document_frequency),
            (time.perf_counter() - started) * 1000,
        )
        return index

    def invalidate(self, tenant_id: Optional[str] = None) -> None:
        """Drop cached indexes after ingestion changes the corpus."""
        if tenant_id:
            self._indexes.pop(tenant_id, None)
        else:
            self._indexes.clear()

    # ------------------------------------------------------------------ search
    async def search(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 20,
        boost_phrases: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        """Rank chunks by BM25, boosting exact phrase matches."""
        index = await self.build_index(tenant_id)
        if not index.documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored = self._score(index, query_terms)

        # A quoted phrase that appears verbatim is stronger evidence than any
        # bag-of-words score, so it lifts the document above term-frequency ties.
        if boost_phrases:
            lowered = [p.lower() for p in boost_phrases if p]
            for position, (document, score) in enumerate(scored):
                text_lower = document.text.lower()
                hits = sum(1 for phrase in lowered if phrase in text_lower)
                if hits:
                    scored[position] = (document, score * (1.0 + 0.5 * hits))

        scored.sort(key=lambda pair: pair[1], reverse=True)

        results: List[RetrievedChunk] = []
        best = scored[0][1] if scored else 1.0
        for rank, (document, score) in enumerate(scored[:top_k], start=1):
            if score <= 0:
                break
            results.append(
                RetrievedChunk(
                    chunk_id=document.chunk_id,
                    text=document.text,
                    parent_doc_id=document.parent_doc_id,
                    # Normalized against the top hit so scores are comparable
                    # across queries; BM25 has no fixed upper bound.
                    score=round(score / best, 4) if best else 0.0,
                    section_path=document.section_path,
                    retrieval_path="lexical",
                    rank=rank,
                )
            )
        return results

    def _score(
        self, index: BM25Index, query_terms: Sequence[str]
    ) -> List[Tuple[_Document, float]]:
        """Okapi BM25 over the tenant's documents."""
        total_documents = index.size
        idf_cache: Dict[str, float] = {}

        for term in set(query_terms):
            document_frequency = index.document_frequency.get(term, 0)
            if document_frequency == 0:
                idf_cache[term] = 0.0
                continue
            # The +0.5 smoothing keeps IDF positive for terms in most documents.
            idf_cache[term] = math.log(
                1 + (total_documents - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

        scored: List[Tuple[_Document, float]] = []
        for document in index.documents:
            score = 0.0
            for term in query_terms:
                frequency = document.term_frequencies.get(term)
                if not frequency:
                    continue
                idf = idf_cache.get(term, 0.0)
                if idf <= 0:
                    continue
                normalization = self.K1 * (
                    1 - self.B + self.B * (document.length / (index.average_length or 1))
                )
                score += idf * (frequency * (self.K1 + 1)) / (frequency + normalization)
            if score > 0:
                scored.append((document, score))
        return scored

    def stats(self, tenant_id: str) -> Dict[str, object]:
        index = self._indexes.get(tenant_id)
        if not index:
            return {"indexed": False}
        return {
            "indexed": True,
            "chunks": index.size,
            "terms": len(index.document_frequency),
            "average_length": round(index.average_length, 1),
            "age_seconds": round(time.time() - index.built_at, 1),
        }


lexical_search_service = LexicalSearchService()
