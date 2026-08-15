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

import asyncio
import hashlib
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

from app.core.config import settings
from app.core.exceptions import ModelUnavailableError

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Shared across the embedding and reranking services: both run torch inference,
# and a single bounded pool keeps total CPU contention predictable. Bounded
# because unbounded threads would oversubscribe the CPU and make every request
# slower rather than fewer requests fast.
_INFERENCE_POOL = ThreadPoolExecutor(
    max_workers=settings.INFERENCE_THREADS, thread_name_prefix="inference"
)


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
    def embedding_version(self) -> str:
        """Identity stamped onto every vector this service writes.

        Vectors from different models occupy different spaces, so comparing them
        produces plausible-looking similarity scores that are meaningless. The
        version travels with the data so a mismatch is detectable rather than
        silently degrading retrieval quality.
        """
        if self.is_semantic:
            return settings.EMBEDDING_VERSION
        return f"lexical-hash/{settings.EMBEDDING_DIMENSIONS}/v1"

    def is_compatible(self, stored_version: Optional[str]) -> bool:
        """Whether a stored vector was written by the current model."""
        if not settings.STRICT_EMBEDDING_VERSION:
            return True
        if not stored_version:
            # Vectors predating version stamping. Treated as incompatible under
            # strict mode: an unlabelled vector cannot be shown to match.
            return False
        return stored_version == self.embedding_version

    @property
    def dimensions(self) -> int:
        if self._model is not None:
            # The accessor was renamed across sentence-transformers versions; try
            # the current name first so newer installs do not emit a deprecation
            # warning on every call.
            for accessor in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                getter = getattr(self._model, accessor, None)
                if getter is None:
                    continue
                try:
                    return int(getter())
                except Exception:  # noqa: BLE001
                    continue
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

    # ------------------------------------------------------------------ async
    async def encode_batch_async(self, texts: Sequence[str]) -> List[List[float]]:
        """Encode off the event loop.

        Transformer inference is synchronous CPU work. Called directly from an
        async handler it blocks the loop for its whole duration, so every other
        in-flight request stalls behind it — a request that only needed a fast
        graph lookup waits on someone else's embedding. Running it in a worker
        thread lets the loop keep serving.

        The GIL is released inside torch's compute kernels, so threads give real
        parallelism here rather than the illusion of it.
        """
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _INFERENCE_POOL, self.encode_batch, list(texts)
        )

    async def encode_query_async(self, query: str) -> List[float]:
        """Encode a search query off the event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_INFERENCE_POOL, self.encode_query, query)

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
            "model_id": settings.EMBEDDING_MODEL,
            "embedding_version": self.embedding_version,
            "dimensions": self.dimensions,
            "strict_version_check": settings.STRICT_EMBEDDING_VERSION,
            "fallback_reason": self._load_failed_reason,
        }


embedding_service = EmbeddingService()
