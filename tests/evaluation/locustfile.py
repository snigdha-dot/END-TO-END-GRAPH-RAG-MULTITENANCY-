"""Locust Performance & Load Testing Suite for Team B Graph RAG API."""
import random
from locust import HttpUser, task, between

class GraphRAGUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def test_retrieval_search(self):
        """Simulate high-frequency retrieval search queries from Team A Chatbots."""
        queries = [
            "What services fail if Auth Service goes down?",
            "Who owns the Payment Gateway system?",
            "What microservices depend on User Database?",
            "Show me all critical components in production environment.",
            "Who manages the platform team?"
        ]
        tenants = ["tech_support_bot", "hr_support_bot", "legal_bot"]

        tenant = random.choice(tenants)
        query = random.choice(queries)

        payload = {
            "tenant_id": tenant,
            "user_query": query,
            "options": {
                "max_traversal_depth": 2,
                "top_k": 5
            }
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": "team_a_secret_key_123",
            "X-Tenant-ID": tenant
        }

        with self.client.post("/api/v1/retrieval/search", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "telemetry" in data and "subgraph" in data:
                    response.success()
                else:
                    response.failure("Response missing telemetry or subgraph fields")
            else:
                response.failure(f"HTTP Status {response.status_code}: {response.text}")

    @task(1)
    def test_health_check(self):
        """Simulate load balancer health check pings."""
        self.client.get("/health")
