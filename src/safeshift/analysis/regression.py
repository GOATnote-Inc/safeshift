"""Regression testing — CI-friendly, exit code 1 on safety regression."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from safeshift.grader import GradeResult
from safeshift.thresholds import REGRESSION

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionResult:
    """Result of a regression comparison."""

    passed: bool
    baseline_mean_safety: float
    current_mean_safety: float
    delta: float
    threshold: float
    new_class_a_failures: list[str]  # scenario IDs with new Class A
    message: str


def load_latencies(path: str | Path) -> dict[str, float]:
    """Load mean latency per optimization from results.jsonl.

    Returns dict of optimization -> mean latency_ms.
    Returns empty dict if file is missing.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("Results file not found: %s — using empty latencies", path)
        return {}

    latency_sums: dict[str, float] = {}
    latency_counts: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            opt = entry.get("optimization", "")
            latency = entry.get("latency_ms")
            if opt and latency is not None:
                latency_sums[opt] = latency_sums.get(opt, 0.0) + latency
                latency_counts[opt] = latency_counts.get(opt, 0) + 1

    return {opt: latency_sums[opt] / latency_counts[opt] for opt in latency_sums}


def load_grades(path: str | Path) -> list[GradeResult]:
    """Load grade results from a JSONL file."""
    path = Path(path)
    grades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                grades.append(GradeResult.from_dict(json.loads(line)))
    return grades


def run_regression(
    baseline_path: str | Path,
    current_path: str | Path,
    threshold: float = REGRESSION.safety_drop,
) -> RegressionResult:
    """Compare current grades against baseline.

    Returns RegressionResult. Fails if:
    - Mean safety drops by more than threshold
    - Any new Class A failures appear

    Args:
        baseline_path: Path to baseline grades.jsonl
        current_path: Path to current grades.jsonl
        threshold: Maximum allowed safety score drop
    """
    baseline_grades = load_grades(baseline_path)
    current_grades = load_grades(current_path)

    baseline_safety = [g.safety_score for g in baseline_grades]
    current_safety = [g.safety_score for g in current_grades]

    mean_baseline = sum(baseline_safety) / len(baseline_safety) if baseline_safety else 0.0
    mean_current = sum(current_safety) / len(current_safety) if current_safety else 0.0
    delta = mean_current - mean_baseline

    # Check for new Class A failures
    baseline_class_a = {g.scenario_id for g in baseline_grades if g.failure_class.value == "A"}
    current_class_a = {g.scenario_id for g in current_grades if g.failure_class.value == "A"}
    new_class_a = sorted(current_class_a - baseline_class_a)

    # Determine pass/fail
    safety_regression = delta < -threshold
    new_failures = len(new_class_a) > 0
    passed = not safety_regression and not new_failures

    reasons = []
    if safety_regression:
        reasons.append(f"safety regression: {delta:+.4f} exceeds threshold {threshold}")
    if new_failures:
        reasons.append(f"new Class A failures: {new_class_a}")

    message = "PASS: no regression detected" if passed else f"FAIL: {'; '.join(reasons)}"

    return RegressionResult(
        passed=passed,
        baseline_mean_safety=round(mean_baseline, 4),
        current_mean_safety=round(mean_current, 4),
        delta=round(delta, 4),
        threshold=threshold,
        new_class_a_failures=new_class_a,
        message=message,
    )
