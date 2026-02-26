"""Executor — pluggable inference backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutorResult:
    """Result from a single inference execution."""

    response_text: str
    latency_ms: float
    ttft_ms: float | None = None  # Time to first token
    tokens_per_sec: float | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    gpu_memory_mb: float | None = None
    batch_position: int | None = None
    model: str = ""
    optimization: str = "baseline"
    scenario_id: str = ""
    executor_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_text": self.response_text,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "gpu_memory_mb": self.gpu_memory_mb,
            "batch_position": self.batch_position,
            "model": self.model,
            "optimization": self.optimization,
            "scenario_id": self.scenario_id,
            "executor_type": self.executor_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutorResult:
        return cls(
            response_text=data["response_text"],
            latency_ms=data["latency_ms"],
            ttft_ms=data.get("ttft_ms"),
            tokens_per_sec=data.get("tokens_per_sec"),
            total_tokens=data.get("total_tokens"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            gpu_memory_mb=data.get("gpu_memory_mb"),
            batch_position=data.get("batch_position"),
            model=data.get("model", ""),
            optimization=data.get("optimization", "baseline"),
            scenario_id=data.get("scenario_id", ""),
            executor_type=data.get("executor_type", ""),
            metadata=data.get("metadata", {}),
        )


class Executor(ABC):
    """Abstract inference backend."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Executor type name."""

    @property
    def supports_optimization(self) -> bool:
        """Whether this executor supports optimization configs."""
        return False

    @abstractmethod
    async def execute(
        self,
        messages: list[dict[str, str]],
        model: str,
        optimization: str = "baseline",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 4096,
    ) -> ExecutorResult:
        """Execute inference and return result with timing."""

    async def health_check(self) -> bool:
        """Check if the executor backend is available."""
        return True

    async def close(self) -> None:
        """Clean up resources."""
