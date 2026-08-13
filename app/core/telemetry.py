"""Telemetry, Latency, and Side-by-Side Model Cost Tracker."""
import time
from typing import Dict, List, Any
from app.core.config import settings

class ModelCallMetrics:
    def __init__(self, step_name: str, model_name: str, prompt_tokens: int = 0, completion_tokens: int = 0, latency_ms: float = 0.0):
        self.step_name = step_name
        self.model_name = model_name
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = round(latency_ms, 2)
        self.cost_usd = self._calculate_cost()

    def _calculate_cost(self) -> float:
        pricing = settings.MODEL_PRICING.get(self.model_name, {"input": 0.0, "output": 0.0})
        input_cost = self.prompt_tokens * pricing["input"]
        output_cost = self.completion_tokens * pricing["output"]
        return round(input_cost + output_cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_name,
            "model_name": self.model_name,
            "tokens_used": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens
            },
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms
        }


class TelemetryTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.latency_breakdown_ms: Dict[str, float] = {}
        self.model_calls: List[ModelCallMetrics] = []

    def record_step_latency(self, step_name: str, duration_ms: float):
        """Record latency duration for a pipeline step."""
        self.latency_breakdown_ms[step_name] = round(duration_ms, 2)

    def record_model_call(self, step_name: str, model_name: str, prompt_tokens: int, completion_tokens: int, duration_ms: float):
        """Record model execution token usage, cost, and latency."""
        metric = ModelCallMetrics(
            step_name=step_name,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=duration_ms
        )
        self.model_calls.append(metric)

    def finalize(self) -> Dict[str, Any]:
        total_latency = round((time.perf_counter() - self.start_time) * 1000, 2)
        self.latency_breakdown_ms["total_retrieval_latency"] = total_latency
        
        total_cost = round(sum(m.cost_usd for m in self.model_calls), 6)

        return {
            "latency_breakdown_ms": self.latency_breakdown_ms,
            "model_cost_breakdown": {
                "models_called": [m.to_dict() for m in self.model_calls],
                "total_request_cost_usd": total_cost
            }
        }
