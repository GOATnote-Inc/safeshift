"""Scenario — safety-critical decision under constraint."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_CHECK_TYPES = {"regex", "contains", "not_contains", "judge"}
VALID_TIME_PRESSURES = {"immediate", "minutes", "hours"}
VALID_DOMAINS = {"clinical", "robotic", "clinical_defer", "robotic_defer"}


def _require(data: dict, key: str, context: str) -> Any:
    """Get a required key from a dict, raising ValueError with context on missing."""
    if key not in data:
        raise ValueError(f"Missing required field '{key}' in {context}")
    return data[key]


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
    """Load a scenario from a YAML file.

    Raises ValueError with actionable diagnostics on missing/invalid fields.
    """
    path = Path(path)
    try:
        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML did not parse to a dict")

        # Top-level required fields
        scenario_id = _require(data, "id", f"scenario ({path})")
        name = _require(data, "name", f"scenario ({path})")
        domain = _require(data, "domain", f"scenario ({path})")
        description = _require(data, "description", f"scenario ({path})")
        expected_action = _require(data, "expected_action", f"scenario ({path})")

        # Domain validation
        if domain not in VALID_DOMAINS:
            raise ValueError(
                f"Invalid domain '{domain}' in scenario ({path}). "
                f"Must be one of: {sorted(VALID_DOMAINS)}"
            )

        # Messages validation
        messages = _require(data, "messages", f"scenario ({path})")
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError(f"'messages' must be a non-empty list in scenario ({path})")
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"messages[{i}] must be a dict in scenario ({path})")
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    f"messages[{i}] must have 'role' and 'content' keys in scenario ({path})"
                )

        # Latency budget
        lb_data = _require(data, "latency_budget", f"scenario ({path})")
        latency = LatencyBudget(
            target_ms=_require(lb_data, "target_ms", f"latency_budget in scenario ({path})"),
            acceptable_ms=_require(
                lb_data, "acceptable_ms", f"latency_budget in scenario ({path})"
            ),
            critical_ms=_require(lb_data, "critical_ms", f"latency_budget in scenario ({path})"),
        )

        # Validate latency values are numeric
        for field_name in ("target_ms", "acceptable_ms", "critical_ms"):
            val = getattr(latency, field_name)
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"latency_budget.{field_name} must be numeric, "
                    f"got {type(val).__name__} in scenario ({path})"
                )

        # Safety invariants
        invariants = []
        for i, inv in enumerate(data.get("safety_invariants", [])):
            inv_ctx = f"safety_invariants[{i}] in scenario ({path})"
            inv_name = _require(inv, "name", inv_ctx)
            inv_desc = _require(inv, "description", inv_ctx)
            check_type = _require(inv, "check_type", inv_ctx)

            if check_type not in VALID_CHECK_TYPES:
                raise ValueError(
                    f"Invalid check_type '{check_type}' in {inv_ctx}. "
                    f"Must be one of: {sorted(VALID_CHECK_TYPES)}"
                )

            severity = inv.get("severity", 1.0)
            if not isinstance(severity, (int, float)):
                raise ValueError(f"severity must be numeric in {inv_ctx}")

            invariants.append(
                SafetyInvariant(
                    name=inv_name,
                    description=inv_desc,
                    check_type=check_type,
                    pattern=inv.get("pattern"),
                    judge_criterion=inv.get("judge_criterion"),
                    severity=severity,
                )
            )

        # Constraint
        constraint_data = _require(data, "constraint", f"scenario ({path})")
        time_pressure = _require(
            constraint_data, "time_pressure", f"constraint in scenario ({path})"
        )
        if time_pressure not in VALID_TIME_PRESSURES:
            raise ValueError(
                f"Invalid time_pressure '{time_pressure}' in constraint in scenario ({path}). "
                f"Must be one of: {sorted(VALID_TIME_PRESSURES)}"
            )

        constraint = ConstraintSpec(
            description=_require(
                constraint_data, "description", f"constraint in scenario ({path})"
            ),
            time_pressure=time_pressure,
            consequence_of_delay=_require(
                constraint_data, "consequence_of_delay", f"constraint in scenario ({path})"
            ),
        )

        # Severity validation
        severity = data.get("severity", 1.0)
        if not isinstance(severity, (int, float)):
            raise ValueError(f"severity must be numeric in scenario ({path})")

        return Scenario(
            id=scenario_id,
            name=name,
            domain=domain,
            description=description,
            messages=messages,
            latency_budget=latency,
            safety_invariants=invariants,
            constraint=constraint,
            rubric=data.get("rubric", "rubric_clinical"),
            expected_action=expected_action,
            severity=severity,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to load scenario from {path}: {e}") from e


def load_scenarios_from_dir(directory: str | Path) -> list[Scenario]:
    """Load all scenarios from a directory (recursive)."""
    directory = Path(directory)
    scenarios = []
    for path in sorted(directory.rglob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios
