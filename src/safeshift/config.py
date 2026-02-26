"""RunConfig, MatrixConfig — frozen configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a single evaluation run."""

    scenario_ids: list[str] = field(default_factory=list)
    optimization_ids: list[str] = field(default_factory=list)
    executor: str = "mock"
    model: str = "mock-model"
    judge_model: str = "gpt-4o"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096
    n_trials: int = 1
    output_dir: str = "results"
    remote: str | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.temperature != 0.0:
            errors.append("temperature must be 0.0 for deterministic evaluation")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        return errors


@dataclass(frozen=True)
class MatrixConfig:
    """N scenarios x M optimizations evaluation matrix."""

    name: str
    description: str
    scenario_paths: list[str]
    optimization_paths: list[str]
    executor: str = "mock"
    executor_config_path: str | None = None
    model: str = "mock-model"
    judge_model: str = "gpt-4o"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096
    n_trials: int = 1
    output_dir: str = "results"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errors = []
        if self.temperature != 0.0:
            errors.append("temperature must be 0.0 for deterministic evaluation")
        if not self.scenario_paths:
            errors.append("at least one scenario path required")
        if not self.optimization_paths:
            errors.append("at least one optimization path required")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        return errors


def load_matrix_config(path: str | Path) -> MatrixConfig:
    """Load a matrix configuration from YAML."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    return MatrixConfig(
        name=data["name"],
        description=data.get("description", ""),
        scenario_paths=data["scenario_paths"],
        optimization_paths=data["optimization_paths"],
        executor=data.get("executor", "mock"),
        executor_config_path=data.get("executor_config_path"),
        model=data.get("model", "mock-model"),
        judge_model=data.get("judge_model", "gpt-4o"),
        temperature=data.get("temperature", 0.0),
        seed=data.get("seed", 42),
        max_tokens=data.get("max_tokens", 4096),
        n_trials=data.get("n_trials", 1),
        output_dir=data.get("output_dir", "results"),
        metadata=data.get("metadata", {}),
    )
