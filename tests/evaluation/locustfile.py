"""Locust load-testing suite for the Team B Graph RAG API.

    locust -f tests/evaluation/locustfile.py --host http://localhost:8000

Each simulated user carries one chatbot's credential, mirroring production: a key
is bound to a single tenant, so load is generated per-tenant rather than by one
key fanning across knowledge bases.
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

MOVIES_KEY = os.getenv("LOAD_TEST_MOVIES_KEY", "dev_movies_key_change_me")
AI_TRENDS_KEY = os.getenv("LOAD_TEST_AI_KEY", "dev_ai_trends_key_change_me")

MOVIE_QUERIES = [
    "Which other films did the director of Inception make?",
    "Who starred in Interstellar?",
    "What genre is Dunkirk?",
    "Which studio produced Inception?",
    "Who composed the score for Interstellar?",
]

AI_QUERIES = [
    "What models build on the Transformer architecture?",
    "Which organization released GPT-4?",
    "What datasets was the model trained on?",
    "Which techniques does retrieval augmented generation use?",
    "What benchmarks evaluate reasoning?",
]


class _BaseChatbotUser(HttpUser):
    """Shared behaviour; subclasses bind a tenant credential and query set."""

    abstract = True
    wait_time = between(0.1, 0.5)
    api_key: str = ""
    queries: list[str] = []

    @property
    def headers(self) -> dict:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    @task(10)
    def retrieval_search(self) -> None:
        payload = {
            "user_query": random.choice(self.queries),
            "options": {"max_traversal_depth": 2, "top_k": 5},
        }
        with self.client.post(
            "/api/v1/retrieval/search",
            json=payload,
            headers=self.headers,
            catch_response=True,
            name="/api/v1/retrieval/search",
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if "telemetry" in body and "context_passages" in body:
                    response.success()
                else:
                    response.failure("Response is missing telemetry or context_passages")
            elif response.status_code == 429:
                # Rate limiting working as designed, not a service failure.
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @task(1)
    def health_probe(self) -> None:
        self.client.get("/health", name="/health")

    @task(1)
    def cross_tenant_attempt_must_fail(self) -> None:
        """Continuously assert that isolation holds under load, not just at rest."""
        foreign = "ai_trends_bot" if self.api_key == MOVIES_KEY else "movies_bot"
        with self.client.post(
            "/api/v1/retrieval/search",
            json={"user_query": "cross tenant probe"},
            headers={**self.headers, "X-Tenant-ID": foreign},
            catch_response=True,
            name="/api/v1/retrieval/search [isolation probe]",
        ) as response:
            if response.status_code == 403:
                response.success()
            else:
                response.failure(
                    f"TENANT ISOLATION BREACH: expected 403, got {response.status_code}"
                )


class MoviesChatbotUser(_BaseChatbotUser):
    weight = 1
    api_key = MOVIES_KEY
    queries = MOVIE_QUERIES


class AITrendsChatbotUser(_BaseChatbotUser):
    weight = 1
    api_key = AI_TRENDS_KEY
    queries = AI_QUERIES
