"""Dense embedding service (bge-small-en-v1.5) with a deterministic fallback.

The model is loaded lazily on first use — importing `sentence_transformers` costs
seconds and hundreds of MB, which would make process startup and unit tests slow.

If the model is unavailable (weights not downloaded, torch absent, offline host),
the service degrades to a deterministic hashing embedding rather than failing the
request. That keeps the pipeline exercisable end-to-end without model weights, and
`is_semantic` reports which mode produced a vector so telemetry never claims
semantic quality it did not deliver.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from typing import List, Optional, Sequence

from app.core.config import settings
from app.core.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingService:
    """Encodes text into unit-normalized dense vectors."""

    def __init__(self) -> None:
        self._model = None
        self._load_attempted = False
        self._load_failed_reason: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ loading
    def _load_model(self) -> None:
        """Attempt to load the sentence-transformer model exactly once."""
        if self._load_attempted:
            return
        with self._lock:
            if self._load_attempted:
                return
            self._load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                logger.info("Loading embedding model '%s'...", settings.EMBEDDING_MODEL)
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info("Embedding model loaded (dim=%d).", self.dimensions)
            except Exception as exc:  # noqa: BLE001 - any failure means "degrade"
                self._load_failed_reason = str(exc)
                logger.warning(
                    "Embedding model unavailable (%s). Falling back to lexical hashing "
                    "embeddings; semantic recall will be reduced.", exc,
                )
                if not settings.ALLOW_MODEL_FALLBACK:
                    raise ModelUnavailableError(
                        f"Embedding model '{settings.EMBEDDING_MODEL}' could not be loaded.",
                        reason=str(exc),
                    ) from exc

    @property
    def is_semantic(self) -> bool:
        """True when a real transformer produced the vectors."""
        self._load_model()
        return self._model is not None

    @property
    def model_label(self) -> str:
        """Telemetry label reflecting what actually ran."""
        return settings.EMBEDDING_MODEL_LABEL if self.is_semantic else "lexical-fallback"

    @property
    def dimensions(self) -> int:
        if self._model is not None:
            try:
                return int(self._model.get_sentence_embedding_dimension())
            except Exception:  # noqa: BLE001
                pass
        return settings.EMBEDDING_DIMENSIONS

    # ------------------------------------------------------------------ fallback
    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic bag-of-words hashing embedding, L2-normalized.

        Not semantic: it captures lexical overlap only. Sufficient to keep vector
        search, HNSW indexing, and RRF wiring testable without model weights.
        """
        dim = settings.EMBEDDING_DIMENSIONS
        vec = [0.0] * dim
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign
            # A second bucket reduces collision loss at negligible cost.
            idx2 = int.from_bytes(digest[5:9], "big") % dim
            vec[idx2] += sign * 0.5

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ------------------------------------------------------------------ encoding
    def encode(self, text: str) -> List[float]:
        """Encode a single string into a unit-normalized vector."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode many strings, batching through the model when available."""
        if not texts:
            return []
        self._load_model()

        if self._model is None:
            return [self._hash_embed(t) for t in texts]

        try:
            vectors = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [[float(x) for x in row] for row in vectors]
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding inference failed (%s); using lexical fallback.", exc)
            return [self._hash_embed(t) for t in texts]

    def encode_query(self, query: str) -> List[float]:
        """Encode a search query.

        BGE models are trained with an asymmetric retrieval prefix on the query side;
        omitting it measurably degrades recall.
        """
        self._load_model()
        if self._model is not None and "bge" in settings.EMBEDDING_MODEL.lower():
            prefixed = f"Represent this sentence for searching relevant passages: {query}"
            return self.encode_batch([prefixed])[0]
        return self.encode(query)

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        """Cosine similarity; inputs are expected unit-normalized but not assumed."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def health(self) -> dict:
        return {
            "semantic": self.is_semantic,
            "model": self.model_label,
            "dimensions": self.dimensions,
            "fallback_reason": self._load_failed_reason,
        }


embedding_service = EmbeddingService()
