"""Tests for statistical utilities."""

from safeshift.analysis.statistics import (
    bootstrap_ci,
    cohens_d,
    wilson_score,
)


class TestWilsonScore:
    def test_perfect(self):
        ci = wilson_score(100, 100)
        assert ci.proportion == 1.0
        assert ci.lower > 0.95
        assert ci.upper == 1.0

    def test_zero(self):
        ci = wilson_score(0, 100)
        assert ci.proportion == 0.0
        assert ci.lower == 0.0
        assert ci.upper < 0.05

    def test_half(self):
        ci = wilson_score(50, 100)
        assert 0.39 < ci.lower < 0.41
        assert 0.59 < ci.upper < 0.61

    def test_empty(self):
        ci = wilson_score(0, 0)
        assert ci.proportion == 0.0
        assert ci.n == 0

    def test_small_n(self):
        ci = wilson_score(3, 5)
        assert 0.2 < ci.lower < 0.6
        assert ci.proportion == 0.6

    def test_bounds(self):
        ci = wilson_score(10, 10)
        assert 0.0 <= ci.lower <= ci.proportion <= ci.upper <= 1.0


class TestBootstrapCI:
    def test_basic(self):
        values = [0.8, 0.85, 0.9, 0.75, 0.82]
        ci = bootstrap_ci(values)
        assert 0.75 < ci.mean < 0.90
        assert ci.lower <= ci.mean <= ci.upper
        assert ci.n == 5

    def test_empty(self):
        ci = bootstrap_ci([])
        assert ci.mean == 0.0

    def test_deterministic(self):
        values = [0.5, 0.6, 0.7, 0.8]
        ci1 = bootstrap_ci(values, seed=42)
        ci2 = bootstrap_ci(values, seed=42)
        assert ci1.mean == ci2.mean
        assert ci1.lower == ci2.lower

    def test_tight_ci_large_n(self):
        values = [0.5] * 1000
        ci = bootstrap_ci(values)
        assert abs(ci.upper - ci.lower) < 0.01


class TestCohensD:
    def test_identical_groups(self):
        a = [0.5, 0.5, 0.5]
        b = [0.5, 0.5, 0.5]
        result = cohens_d(a, b)
        assert result.d == 0.0
        assert result.interpretation == "negligible"

    def test_large_effect(self):
        a = [0.9, 0.85, 0.95, 0.88]
        b = [0.2, 0.25, 0.15, 0.22]
        result = cohens_d(a, b)
        assert abs(result.d) > 0.8
        assert result.interpretation == "large"

    def test_empty_groups(self):
        result = cohens_d([], [1.0])
        assert result.d == 0.0

    def test_direction(self):
        a = [0.9, 0.85]
        b = [0.3, 0.35]
        result = cohens_d(a, b)
        assert result.d > 0  # a > b means positive d
