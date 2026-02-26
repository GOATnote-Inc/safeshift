"""Grader — multi-dimensional rubric assessment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from safeshift.executor import ExecutorResult
from safeshift.scenario import Scenario


class FailureClass(str, Enum):
    """Failure classification."""

    A = "A"  # Critical safety drop
    B = "B"  # Drift from baseline
    C = "C"  # Partial degradation
    D = "D"  # Latency violation
    E = "E"  # Hallucination
    NONE = "none"  # No failure


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single grading dimension."""

    dimension: str  # "safety", "accuracy", "completeness", "timeliness", "specificity"
    score: float  # 0.0-1.0
    weight: float
    evidence: str = ""
    layer: int = 0  # Which grading layer produced this (0=pattern, 1=heuristic, 2=judge)


@dataclass(frozen=True)
class GradeResult:
    """Complete grading result for a single execution."""

    scenario_id: str
    optimization: str
    dimensions: list[DimensionScore]
    failure_class: FailureClass = FailureClass.NONE
    failure_reason: str = ""
    overall_score: float = 0.0
    invariant_violations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def safety_score(self) -> float:
        for d in self.dimensions:
            if d.dimension == "safety":
                return d.score
        return 0.0

    @property
    def passed(self) -> bool:
        return self.failure_class == FailureClass.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "optimization": self.optimization,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "score": d.score,
                    "weight": d.weight,
                    "evidence": d.evidence,
                    "layer": d.layer,
                }
                for d in self.dimensions
            ],
            "failure_class": self.failure_class.value,
            "failure_reason": self.failure_reason,
            "overall_score": self.overall_score,
            "invariant_violations": self.invariant_violations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GradeResult:
        dimensions = [
            DimensionScore(
                dimension=d["dimension"],
                score=d["score"],
                weight=d["weight"],
                evidence=d.get("evidence", ""),
                layer=d.get("layer", 0),
            )
            for d in data.get("dimensions", [])
        ]
        return cls(
            scenario_id=data["scenario_id"],
            optimization=data["optimization"],
            dimensions=dimensions,
            failure_class=FailureClass(data.get("failure_class", "none")),
            failure_reason=data.get("failure_reason", ""),
            overall_score=data.get("overall_score", 0.0),
            invariant_violations=data.get("invariant_violations", []),
            metadata=data.get("metadata", {}),
        )


class Grader(ABC):
    """Abstract grading interface."""

    @abstractmethod
    async def grade(
        self,
        scenario: Scenario,
        result: ExecutorResult,
    ) -> GradeResult:
        """Grade a single execution result."""

    async def grade_batch(
        self,
        scenario: Scenario,
        results: list[ExecutorResult],
    ) -> list[GradeResult]:
        """Grade multiple results for the same scenario."""
        grades = []
        for result in results:
            grades.append(await self.grade(scenario, result))
        return grades
