"""RunConfig, MatrixConfig — frozen configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require(data: dict, key: str, context: str) -> Any:
    """Get a required key from a dict, raising ValueError with context on missing."""
    if key not in data:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return data[key]


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
    """Load a matrix configuration from YAML.

    Raises ValueError with actionable diagnostics on missing/invalid fields.
    """
    path = Path(path)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML did not parse to a dict")

        ctx = f"matrix config ({path})"
        name = _require(data, "name", ctx)
        scenario_paths = _require(data, "scenario_paths", ctx)
        optimization_paths = _require(data, "optimization_paths", ctx)

        # Type validation
        temperature = data.get("temperature", 0.0)
        if not isinstance(temperature, (int, float)):
            raise ValueError(
                f"temperature must be numeric in {ctx}, got {type(temperature).__name__}"
            )

        seed = data.get("seed", 42)
        if not isinstance(seed, int):
            raise ValueError(f"seed must be an integer in {ctx}, got {type(seed).__name__}")

        n_trials = data.get("n_trials", 1)
        if not isinstance(n_trials, int):
            raise ValueError(f"n_trials must be an integer in {ctx}, got {type(n_trials).__name__}")

        return MatrixConfig(
            name=name,
            description=data.get("description", ""),
            scenario_paths=scenario_paths,
            optimization_paths=optimization_paths,
            executor=data.get("executor", "mock"),
            executor_config_path=data.get("executor_config_path"),
            model=data.get("model", "mock-model"),
            judge_model=data.get("judge_model", "gpt-4o"),
            temperature=float(temperature),
            seed=seed,
            max_tokens=data.get("max_tokens", 4096),
            n_trials=n_trials,
            output_dir=data.get("output_dir", "results"),
            metadata=data.get("metadata", {}),
        )

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load matrix config from {path}: {e}") from e
