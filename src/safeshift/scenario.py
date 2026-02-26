"""Scenario — safety-critical decision under constraint."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LatencyBudget:
    """Latency thresholds in milliseconds."""

    target_ms: float
    acceptable_ms: float
    critical_ms: float

    def classify(self, latency_ms: float) -> str:
        if latency_ms <= self.target_ms:
            return "target"
        if latency_ms <= self.acceptable_ms:
            return "acceptable"
        if latency_ms <= self.critical_ms:
            return "critical"
        return "violation"


@dataclass(frozen=True)
class SafetyInvariant:
    """A must-hold property for a scenario response."""

    name: str
    description: str
    check_type: str  # "regex", "contains", "not_contains", "judge"
    pattern: str | None = None
    judge_criterion: str | None = None
    severity: float = 1.0  # 0.0-1.0, 1.0 = critical

    def check_deterministic(self, response: str) -> bool | None:
        """Check invariant deterministically. Returns None if judge-only."""
        text = response.lower().strip()
        if self.check_type == "regex":
            return bool(re.search(self.pattern, text, re.IGNORECASE)) if self.pattern else None
        if self.check_type == "contains":
            return self.pattern.lower() in text if self.pattern else None
        if self.check_type == "not_contains":
            return self.pattern.lower() not in text if self.pattern else None
        if self.check_type == "judge":
            return None  # Requires LLM judge
        return None


@dataclass(frozen=True)
class ConstraintSpec:
    """What makes this scenario time-critical."""

    description: str
    time_pressure: str  # "immediate", "minutes", "hours"
    consequence_of_delay: str


@dataclass(frozen=True)
class Scenario:
    """A safety-critical decision scenario for evaluation."""

    id: str
    name: str
    domain: str  # "clinical", "robotic", etc.
    description: str
    messages: list[dict[str, str]]
    latency_budget: LatencyBudget
    safety_invariants: list[SafetyInvariant]
    constraint: ConstraintSpec
    rubric: str  # Reference to rubric YAML (e.g., "rubric_clinical")
    expected_action: str  # What the correct response should do
    severity: float = 1.0  # 0.0-1.0, overall scenario severity
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    latency = LatencyBudget(
        target_ms=data["latency_budget"]["target_ms"],
        acceptable_ms=data["latency_budget"]["acceptable_ms"],
        critical_ms=data["latency_budget"]["critical_ms"],
    )

    invariants = [
        SafetyInvariant(
            name=inv["name"],
            description=inv["description"],
            check_type=inv["check_type"],
            pattern=inv.get("pattern"),
            judge_criterion=inv.get("judge_criterion"),
            severity=inv.get("severity", 1.0),
        )
        for inv in data.get("safety_invariants", [])
    ]

    constraint = ConstraintSpec(
        description=data["constraint"]["description"],
        time_pressure=data["constraint"]["time_pressure"],
        consequence_of_delay=data["constraint"]["consequence_of_delay"],
    )

    messages = data["messages"]

    return Scenario(
        id=data["id"],
        name=data["name"],
        domain=data["domain"],
        description=data["description"],
        messages=messages,
        latency_budget=latency,
        safety_invariants=invariants,
        constraint=constraint,
        rubric=data.get("rubric", "rubric_clinical"),
        expected_action=data["expected_action"],
        severity=data.get("severity", 1.0),
        tags=data.get("tags", []),
        metadata=data.get("metadata", {}),
    )


def load_scenarios_from_dir(directory: str | Path) -> list[Scenario]:
    """Load all scenarios from a directory (recursive)."""
    directory = Path(directory)
    scenarios = []
    for path in sorted(directory.rglob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios
