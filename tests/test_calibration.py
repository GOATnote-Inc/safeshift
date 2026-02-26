"""Tests for inter-grader agreement metrics."""

from safeshift.grader import DimensionScore, GradeResult
from safeshift.grading.calibration import cohens_kappa, compute_agreement


def _make_grade(overall: float) -> GradeResult:
    return GradeResult(
        scenario_id="test",
        optimization="baseline",
        dimensions=[DimensionScore("safety", overall, 0.4)],
        overall_score=overall,
    )


class TestCohensKappa:
    def test_perfect_agreement(self):
        a = [0, 1, 0, 1, 0]
        b = [0, 1, 0, 1, 0]
        assert cohens_kappa(a, b) == 1.0

    def test_no_agreement(self):
        a = [0, 0, 0, 0, 0]
        b = [1, 1, 1, 1, 1]
        k = cohens_kappa(a, b)
        assert k <= 0

    def test_partial_agreement(self):
        a = [0, 1, 0, 1, 1]
        b = [0, 1, 1, 1, 0]
        k = cohens_kappa(a, b)
        assert -1 <= k <= 1

    def test_empty(self):
        assert cohens_kappa([], []) == 0.0


class TestComputeAgreement:
    def test_perfect(self):
        grades_a = [_make_grade(0.9), _make_grade(0.8)]
        grades_b = [_make_grade(0.85), _make_grade(0.75)]
        result = compute_agreement(grades_a, grades_b)
        assert result.percent_agreement == 1.0  # Both above threshold

    def test_disagreement(self):
        grades_a = [_make_grade(0.9), _make_grade(0.8)]
        grades_b = [_make_grade(0.3), _make_grade(0.2)]
        result = compute_agreement(grades_a, grades_b)
        assert result.percent_agreement < 1.0

    def test_dimension_agreement(self):
        grades_a = [_make_grade(0.9)]
        grades_b = [_make_grade(0.85)]  # Within 0.2 tolerance
        result = compute_agreement(grades_a, grades_b)
        assert result.dimension_agreement.get("safety", 0) == 1.0
