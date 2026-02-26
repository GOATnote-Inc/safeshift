"""OptimizationConfig — what inference optimization we're testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OptimizationAxis:
    """A single axis of optimization."""

    name: str  # "quantization", "batch_size", "speculative", "attention", "kv_cache"
    value: str | int | float
    description: str = ""

    def __str__(self) -> str:
        return f"{self.name}={self.value}"


@dataclass(frozen=True)
class OptimizationConfig:
    """A complete optimization configuration to evaluate."""

    id: str
    name: str
    axes: list[OptimizationAxis]
    is_baseline: bool = False
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.is_baseline:
            return "baseline"
        return "+".join(str(a) for a in self.axes)

    def to_executor_params(self) -> dict[str, Any]:
        """Convert to parameters for executor configuration."""
        params: dict[str, Any] = {}
        for axis in self.axes:
            params[axis.name] = axis.value
        return params


def load_optimizations(path: str | Path) -> list[OptimizationConfig]:
    """Load optimization configs from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    configs = []
    for item in data.get("optimizations", []):
        axes = [
            OptimizationAxis(
                name=a["name"],
                value=a["value"],
                description=a.get("description", ""),
            )
            for a in item.get("axes", [])
        ]
        configs.append(
            OptimizationConfig(
                id=item["id"],
                name=item["name"],
                axes=axes,
                is_baseline=item.get("is_baseline", False),
                description=item.get("description", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return configs
