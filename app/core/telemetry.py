"""Side-by-side latency (ms) and model cost (USD) telemetry (plan section 5).

Every retrieval response and test run carries a per-step latency breakdown and a
per-model token/cost breakdown, so performance bottlenecks and operational spend
are both visible without instrumenting anything downstream.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from app.core.config import settings

# Steps always present in the response, even when a stage is skipped. Team A codes
# against this contract, so the key set must not vary by execution path.
CONTRACT_STEPS = (
    "query_entity_linking",
    "arcadedb_vector_knn",
    "arcadedb_cypher_traversal",
    "rrf_reranking",
)


class ModelCallMetrics:
    """Token usage, cost, and latency for one model invocation."""

    def __init__(
        self,
        step_name: str,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        self.step_name = step_name
        self.model_name = model_name
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.completion_tokens = max(0, int(completion_tokens))
        self.latency_ms = round(latency_ms, 2)
        self.cost_usd = self._calculate_cost()

    def _calculate_cost(self) -> float:
        pricing = settings.MODEL_PRICING.get(self.model_name)
        if pricing is None:
            # Unknown model: report zero rather than guess, but keep it visible
            # in the breakdown so an unpriced model is noticeable.
            return 0.0
        input_cost = self.prompt_tokens * pricing.get("input", 0.0)
        output_cost = self.completion_tokens * pricing.get("output", 0.0)
        return round(input_cost + output_cost, 8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_name,
            "model_name": self.model_name,
            "tokens_used": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "priced": self.model_name in settings.MODEL_PRICING,
        }


class TelemetryTracker:
    """Accumulates per-step latency and per-model cost for a single request."""

    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.latency_breakdown_ms: Dict[str, float] = {}
        self.model_calls: List[ModelCallMetrics] = []

    def record_step_latency(self, step_name: str, duration_ms: float) -> None:
        self.latency_breakdown_ms[step_name] = round(duration_ms, 2)

    @contextmanager
    def time_step(self, step_name: str) -> Iterator[None]:
        """Time a block and record it, including when it raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_step_latency(step_name, (time.perf_counter() - start) * 1000)

    def record_model_call(
        self,
        step_name: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: float,
    ) -> None:
        self.model_calls.append(
            ModelCallMetrics(
                step_name=step_name,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=duration_ms,
            )
        )

    def finalize(self) -> Dict[str, Any]:
        """Close the books and emit the side-by-side payload."""
        total_latency = round((time.perf_counter() - self.start_time) * 1000, 2)

        breakdown: Dict[str, float] = {step: self.latency_breakdown_ms.get(step, 0.0)
                                       for step in CONTRACT_STEPS}
        for step, value in self.latency_breakdown_ms.items():
            if step not in breakdown:
                breakdown[step] = value
        breakdown["total_retrieval_latency"] = total_latency

        total_cost = round(sum(m.cost_usd for m in self.model_calls), 8)
        total_tokens = sum(m.prompt_tokens + m.completion_tokens for m in self.model_calls)

        return {
            "latency_breakdown_ms": breakdown,
            "model_cost_breakdown": {
                "models_called": [m.to_dict() for m in self.model_calls],
                "total_tokens": total_tokens,
                "total_request_cost_usd": total_cost,
            },
        }
