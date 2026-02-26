"""Tests for regression detection."""

import json
from pathlib import Path

from safeshift.analysis.regression import load_grades, run_regression
from safeshift.grader import DimensionScore, FailureClass, GradeResult


def _write_grades(path: Path, grades: list[GradeResult]) -> None:
    with open(path, "w") as f:
        for g in grades:
            f.write(json.dumps(g.to_dict()) + "\n")


def _make_grade(
    scenario_id: str, safety: float, failure_class: FailureClass = FailureClass.NONE
) -> GradeResult:
    return GradeResult(
        scenario_id=scenario_id,
        optimization="baseline",
        dimensions=[DimensionScore("safety", safety, 0.4)],
        failure_class=failure_class,
        overall_score=safety,
    )


class TestRegression:
    def test_no_regression(self, tmp_path):
        baseline = [_make_grade("s1", 0.9), _make_grade("s2", 0.85)]
        current = [_make_grade("s1", 0.88), _make_grade("s2", 0.87)]
        _write_grades(tmp_path / "baseline.jsonl", baseline)
        _write_grades(tmp_path / "current.jsonl", current)

        result = run_regression(tmp_path / "baseline.jsonl", tmp_path / "current.jsonl")
        assert result.passed
        assert "PASS" in result.message

    def test_safety_regression(self, tmp_path):
        baseline = [_make_grade("s1", 0.9), _make_grade("s2", 0.85)]
        current = [_make_grade("s1", 0.6), _make_grade("s2", 0.5)]
        _write_grades(tmp_path / "baseline.jsonl", baseline)
        _write_grades(tmp_path / "current.jsonl", current)

        result = run_regression(tmp_path / "baseline.jsonl", tmp_path / "current.jsonl")
        assert not result.passed
        assert result.delta < 0

    def test_new_class_a(self, tmp_path):
        baseline = [_make_grade("s1", 0.9)]
        current = [_make_grade("s1", 0.1, FailureClass.A)]
        _write_grades(tmp_path / "baseline.jsonl", baseline)
        _write_grades(tmp_path / "current.jsonl", current)

        result = run_regression(tmp_path / "baseline.jsonl", tmp_path / "current.jsonl")
        assert not result.passed
        assert "s1" in result.new_class_a_failures

    def test_existing_class_a_not_flagged(self, tmp_path):
        baseline = [_make_grade("s1", 0.1, FailureClass.A)]
        current = [_make_grade("s1", 0.1, FailureClass.A)]
        _write_grades(tmp_path / "baseline.jsonl", baseline)
        _write_grades(tmp_path / "current.jsonl", current)

        result = run_regression(
            tmp_path / "baseline.jsonl",
            tmp_path / "current.jsonl",
            threshold=1.0,  # Relax threshold
        )
        assert result.new_class_a_failures == []

    def test_load_grades(self, tmp_path):
        grades = [_make_grade("s1", 0.9), _make_grade("s2", 0.8)]
        _write_grades(tmp_path / "test.jsonl", grades)
        loaded = load_grades(tmp_path / "test.jsonl")
        assert len(loaded) == 2
        assert loaded[0].scenario_id == "s1"
