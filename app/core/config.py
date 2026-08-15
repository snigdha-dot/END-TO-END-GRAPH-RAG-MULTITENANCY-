"""Configuration settings for Team B Graph RAG Service.

All secrets are sourced from the environment (or a local .env file). No secret
has a usable default: `validate_production()` refuses to start a production
process that is still holding development placeholders.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Team B Multi-Tenant Graph RAG Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ------------------------------------------------------------------ ArcadeDB
    ARCADEDB_URL: str = "http://localhost:2480"
    ARCADEDB_USER: str = "root"
    ARCADEDB_PASSWORD: str = "playwithdata"
    # Plan section 2.2 mandates a 3000ms query timeout.
    ARCADEDB_QUERY_TIMEOUT_MS: int = 3000
    ARCADEDB_CONNECT_TIMEOUT_MS: int = 2000
    # Schema DDL and index builds are far slower than queries, especially on first
    # creation. They are admin-path operations, so the tight query bound does not apply.
    ARCADEDB_DDL_TIMEOUT_MS: int = 60000
    # Ingestion writes tolerate more than the read path: the Cypher engine has a
    # multi-second first-call warmup, and writes are not on the user-facing path.
    ARCADEDB_WRITE_TIMEOUT_MS: int = 30000
    # Multi-hop traversal on a dense graph legitimately exceeds the 3000ms read
    # bound. It is given its own budget and degrades to vector-only on timeout,
    # rather than failing a request that vector search alone could still answer.
    ARCADEDB_TRAVERSAL_TIMEOUT_MS: int = 8000
    # Ingestion writes are I/O-bound one-statement-per-request round trips. Bounded
    # concurrency overlaps the waiting without exhausting the pool or starving
    # concurrent retrieval traffic.
    # Measured: 16 concurrent writers overwhelmed a single-node ArcadeDB and it
    # returned 503. 6 keeps it responsive while still overlapping most of the
    # round-trip latency.
    ARCADEDB_WRITE_CONCURRENCY: int = 6
    # Entities per chunk. Real prose yields long tails of low-confidence mentions
    # that add writes without adding retrievable signal.
    MAX_ENTITIES_PER_CHUNK: int = 25
    ARCADEDB_MAX_CONNECTIONS: int = 50
    ARCADEDB_MAX_KEEPALIVE: int = 20
    # Ingestion writes are batched into a single scripted request of this size.
    ARCADEDB_WRITE_BATCH_SIZE: int = 100

    # ------------------------------------------------------------------ Auth
    API_KEY_HEADER: str = "X-API-Key"
    TENANT_HEADER: str = "X-Tenant-ID"
    REQUEST_ID_HEADER: str = "X-Request-ID"

    # api_key -> tenant_id. The key *determines* the tenant; a header that
    # disagrees is rejected rather than silently overridden.
    API_KEY_TENANT_MAP: Dict[str, str] = Field(
        default_factory=lambda: {
            "dev_movies_key_change_me": "movies_bot",
            "dev_ai_trends_key_change_me": "ai_trends_bot",
        }
    )

    # Provisioning/admin key. Never issued to Team A; empty disables admin routes.
    ADMIN_API_KEY: str = ""

    # JWT (plan Security Layer 1: "verifies token claims against requested tenant_id")
    JWT_ENABLED: bool = False
    JWT_SECRET: str = "dev_jwt_secret_change_me"
    JWT_ALGORITHM: str = "HS256"
    JWT_ISSUER: str = "team-a-chatbot-platform"
    JWT_AUDIENCE: str = "team-b-retrieval-service"
    JWT_LEEWAY_SECONDS: int = 30
    # When both are enabled a request must satisfy the API key *and* the JWT.
    JWT_REQUIRED: bool = False

    # ------------------------------------------------------------------ Limits
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120
    RATE_LIMIT_BURST: int = 20
    MAX_REQUEST_BODY_BYTES: int = 1_048_576  # 1 MiB

    # Tenant provisioning is an explicit admin action, never a query side effect.
    ALLOW_TENANT_AUTOPROVISION: bool = False

    # ------------------------------------------------------------------ CORS
    # Comma-separated or JSON. Typed as `str` because pydantic-settings JSON-parses
    # complex-typed fields before validators run, which rejects the comma form;
    # `cors_origins` below is the parsed accessor.
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000"
    CORS_ALLOW_CREDENTIALS: bool = False

    # ------------------------------------------------------------------ Retrieval
    DEFAULT_MAX_HOPS: int = 2
    MAX_TRAVERSAL_DEPTH: int = 3
    MAX_TRAVERSAL_NODES: int = 100
    # Row cap for full scans (vector index builds, BM25 index builds, re-indexing).
    # Distinct from MAX_TRAVERSAL_NODES: that bounds a traversal to keep a single
    # query cheap, whereas a scan legitimately reads the whole corpus.
    MAX_SCAN_ROWS: int = 50_000
    # Bounds on the graph -> text bridge. A traversal can reach 100 entities, and
    # fetching every chunk each appears in transfers far more than fusion can use:
    # measured ~4s on a 400-chunk tenant, nearly all of it text that ranks below
    # the top-k and is discarded.
    GRAPH_CHUNK_SEED_LIMIT: int = 25
    GRAPH_CHUNK_LIMIT: int = 30
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.70
    ENTITY_LINK_SIMILARITY: float = 0.85
    EDGE_CONFIDENCE_THRESHOLD: float = 0.80
    RRF_K: int = 60

    # ------------------------------------------------------------------ Models
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_MODEL_LABEL: str = "bge-small-en-v1.5"
    EMBEDDING_DIMENSIONS: int = 384
    # Stamped onto every chunk alongside its vector. Vectors from different models
    # are not comparable, so mixing them silently corrupts similarity scores in a
    # way that looks like poor retrieval rather than a bug. Bump this when the
    # model changes; the retriever then refuses stale vectors instead of scoring
    # them.
    EMBEDDING_VERSION: str = "bge-small-en-v1.5/384/v1"
    CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_LABEL: str = "ms-marco-MiniLM-L-6-v2"
    RERANKER_ENABLED: bool = True
    RERANK_CANDIDATE_MULTIPLIER: int = 4
    # Characters of each passage the cross-encoder sees. Its cost is quadratic in
    # sequence length, and a record chunk carrying 34 CSV columns is mostly fields
    # the query never mentions. Measured at 237ms per full-length candidate.
    RERANK_MAX_CHARS: int = 512
    RERANK_BATCH_SIZE: int = 32
    # Worker threads for model inference. Transformer forward passes are
    # synchronous CPU work; run on the event loop they stall every concurrent
    # request. Bounded so threads do not oversubscribe the CPU and make all
    # requests slower instead of a few fast.
    INFERENCE_THREADS: int = 4
    # How long a tenant's cached vectors stay valid. Refetching every embedding
    # over HTTP costs ~330ms per query against a 400-chunk tenant, of which only
    # ~11ms is the scoring itself. Ingestion invalidates the cache explicitly, so
    # this bound only covers writes made by another process.
    VECTOR_INDEX_TTL_SECONDS: int = 300
    # Degrade to lexical scoring instead of crashing when model weights are absent.
    ALLOW_MODEL_FALLBACK: bool = True
    # Refuse to score vectors written by a different embedding model. Set false
    # only for a deliberate mixed-version read during migration.
    STRICT_EMBEDDING_VERSION: bool = True

    NER_BACKEND: Literal["gliner", "spacy", "regex"] = "regex"
    GLINER_MODEL: str = "urchade/gliner_small-v2.1"
    SPACY_MODEL: str = "en_core_web_sm"

    # LLM extraction/disambiguation is designed for but not enabled (FOSS-only build).
    LLM_PROVIDER: Literal["none", "gemini", "openai"] = "none"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"

    # ------------------------------------------------------------------ Chunking
    CHUNK_TARGET_TOKENS: int = 500
    CHUNK_MIN_TOKENS: int = 400
    CHUNK_MAX_TOKENS: int = 600
    CHUNK_OVERLAP_TOKENS: int = 100

    # ------------------------------------ Side-by-Side Model Cost Matrix (USD/token)
    MODEL_PRICING: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "bge-small-en-v1.5": {"input": 0.0, "output": 0.0},
            "all-MiniLM-L6-v2": {"input": 0.0, "output": 0.0},
            "ms-marco-MiniLM-L-6-v2": {"input": 0.0, "output": 0.0},
            "gliner_small-v2.1": {"input": 0.0, "output": 0.0},
            "spacy-en_core_web_sm": {"input": 0.0, "output": 0.0},
            "lexical-fallback": {"input": 0.0, "output": 0.0},
            "gemini-1.5-flash": {"input": 0.000075 / 1000, "output": 0.00030 / 1000},
            "gemini-1.5-pro": {"input": 0.00125 / 1000, "output": 0.00500 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
        }
    )

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ Validators
    @field_validator("API_KEY_TENANT_MAP", "MODEL_PRICING", mode="before")
    @classmethod
    def _parse_json_mapping(cls, v: Any) -> Any:
        """Accept a JSON string from the environment for dict-valued settings."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return {}
            return json.loads(v)
        return v

    # ------------------------------------------------------------------ Helpers
    @property
    def cors_origins(self) -> List[str]:
        """Allowed CORS origins, from either a JSON array or a comma-separated list."""
        raw = (self.CORS_ALLOW_ORIGINS or "").strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                return [str(o) for o in json.loads(raw)]
            except json.JSONDecodeError:
                return []
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def query_timeout_seconds(self) -> float:
        return self.ARCADEDB_QUERY_TIMEOUT_MS / 1000.0

    @property
    def connect_timeout_seconds(self) -> float:
        return self.ARCADEDB_CONNECT_TIMEOUT_MS / 1000.0

    def tenant_for_api_key(self, api_key: str) -> str | None:
        """Resolve the single tenant an API key is authorised for."""
        return self.API_KEY_TENANT_MAP.get(api_key)

    def validate_production(self) -> list[str]:
        """Return the list of production-readiness violations (empty == safe)."""
        problems: list[str] = []
        if not self.is_production:
            return problems

        placeholder = "change_me"
        if any(placeholder in k for k in self.API_KEY_TENANT_MAP):
            problems.append("API_KEY_TENANT_MAP still contains development placeholder keys.")
        if not self.API_KEY_TENANT_MAP:
            problems.append("API_KEY_TENANT_MAP is empty; no caller could authenticate.")
        if placeholder in self.JWT_SECRET:
            problems.append("JWT_SECRET is still the development placeholder.")
        if self.JWT_ENABLED and len(self.JWT_SECRET) < 32:
            problems.append("JWT_SECRET must be at least 32 characters.")
        if self.ARCADEDB_PASSWORD == "playwithdata":
            problems.append("ARCADEDB_PASSWORD is still the public default.")
        if "*" in self.cors_origins:
            problems.append("CORS_ALLOW_ORIGINS must not be '*' in production.")
        if self.CORS_ALLOW_CREDENTIALS and "*" in self.cors_origins:
            problems.append("CORS_ALLOW_CREDENTIALS cannot be combined with a wildcard origin.")
        if self.ALLOW_TENANT_AUTOPROVISION:
            problems.append("ALLOW_TENANT_AUTOPROVISION must be false in production.")
        if not self.RATE_LIMIT_ENABLED:
            problems.append("RATE_LIMIT_ENABLED must be true in production.")
        for key, tenant in self.API_KEY_TENANT_MAP.items():
            if len(key) < 24:
                problems.append(f"API key for tenant '{tenant}' is shorter than 24 characters.")
        return problems


settings = Settings()
