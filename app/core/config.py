"""Configuration settings for Team B Graph RAG Service."""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Team B Multi-Tenant Graph RAG Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # ArcadeDB Settings
    ARCADEDB_URL: str = os.getenv("ARCADEDB_URL", "http://localhost:2480")
    ARCADEDB_USER: str = os.getenv("ARCADEDB_USER", "root")
    ARCADEDB_PASSWORD: str = os.getenv("ARCADEDB_PASSWORD", "playwithdata")
    
    # API Security
    API_KEY_HEADER: str = "X-API-Key"
    TENANT_HEADER: str = "X-Tenant-ID"
    ALLOWED_API_KEYS: set[str] = {"team_a_secret_key_123", "dev_test_key_456"}
    
    # Retrieval Default Constraints
    DEFAULT_MAX_HOPS: int = 2
    MAX_TRAVERSAL_NODES: int = 100
    DEFAULT_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.70
    
    # Side-by-Side Model Cost Matrix (USD per 1,000 tokens)
    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "bge-small-en-v1.5": {"input": 0.000000, "output": 0.000000},  # Local FOSS model ($0)
        "all-MiniLM-L6-v2": {"input": 0.000000, "output": 0.000000},   # Local FOSS model ($0)
        "gliner-ner-foss": {"input": 0.000000, "output": 0.000000},    # Local FOSS NER ($0)
        "gemini-1.5-flash": {"input": 0.000075 / 1000, "output": 0.00030 / 1000},
        "gemini-1.5-pro": {"input": 0.00125 / 1000, "output": 0.00500 / 1000},
        "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.00060 / 1000}
    }

    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
