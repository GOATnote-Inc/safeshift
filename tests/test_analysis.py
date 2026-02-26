"""Tests for analysis subsystem."""

from safeshift.analysis.degradation import (
    DegradationResult,
    analyze_degradation,
    detect_cliff_edges,
    summarize_failure_classes,
)
from safeshift.analysis.pareto import (
    ParetoPoint,
    build_pareto_points,
    compute_pareto_frontier,
)
from safeshift.grader import DimensionScore, FailureClass, GradeResult


def _make_grade(
    scenario_id: str,
    optimization: str,
    safety: float,
    overall: float = 0.0,
    failure_class: FailureClass = FailureClass.NONE,
) -> GradeResult:
    return GradeResult(
        scenario_id=scenario_id,
        optimization=optimization,
        dimensions=[DimensionScore("safety", safety, 0.4)],
        failure_class=failure_class,
        overall_score=overall or safety,
    )


class TestParetoFrontier:
    def test_single_point(self):
        points = [ParetoPoint("baseline", 0.9, 500.0)]
        result = compute_pareto_frontier(points)
        assert result[0].is_pareto_optimal

    def test_dominated_point(self):
        points = [
            ParetoPoint("baseline", 0.9, 500.0),
            ParetoPoint("fp16", 0.95, 400.0),  # Better safety AND latency
        ]
        result = compute_pareto_frontier(points)
        optimal = [p for p in result if p.is_pareto_optimal]
        dominated = [p for p in result if not p.is_pareto_optimal]
        assert len(optimal) == 1
        assert optimal[0].optimization == "fp16"
        assert len(dominated) == 1

    def test_pareto_tradeoff(self):
        # Both on frontier: one has better safety, other better latency
        points = [
            ParetoPoint("baseline", 0.95, 500.0),
            ParetoPoint("int4", 0.6, 200.0),
        ]
        result = compute_pareto_frontier(points)
        assert all(p.is_pareto_optimal for p in result)

    def test_empty(self):
        assert compute_pareto_frontier([]) == []

    def test_build_pareto_points(self):
        grades = [
            _make_grade("s1", "baseline", 0.9),
            _make_grade("s2", "baseline", 0.8),
            _make_grade("s1", "fp16", 0.85),
            _make_grade("s2", "fp16", 0.7),
        ]
        latencies = {"baseline": 500.0, "fp16": 350.0}
        points = build_pareto_points(grades, latencies)
        assert len(points) == 2
        assert any(p.optimization == "baseline" for p in points)


class TestDegradation:
    def test_no_degradation(self):
        baseline = [_make_grade("s1", "baseline", 0.9)]
        optimized = [_make_grade("s1", "fp16", 0.88)]
        result = analyze_degradation(baseline, optimized, "fp16")
        assert not result.is_cliff_edge
        assert result.delta < 0  # Slight drop

    def test_significant_degradation(self):
        import random

        rng = random.Random(42)
        baseline = [_make_grade(f"s{i}", "baseline", 0.85 + rng.gauss(0, 0.05)) for i in range(10)]
        optimized = [_make_grade(f"s{i}", "int4", 0.25 + rng.gauss(0, 0.05)) for i in range(10)]
        result = analyze_degradation(baseline, optimized, "int4")
        assert result.degraded
        assert result.significant
        assert result.is_cliff_edge

    def test_cliff_edge_detection(self):
        from safeshift.analysis.statistics import EffectSize, WilsonCI

        dr = DegradationResult(
            optimization="int4",
            baseline_safety=0.9,
            optimized_safety=0.3,
            delta=-0.6,
            effect_size=EffectSize(d=-3.0, interpretation="large"),
            wilson_ci=WilsonCI(proportion=0.3, lower=0.2, upper=0.4, n=10),
            is_cliff_edge=True,
            failure_classes={"A": 5},
            n_scenarios=10,
        )
        # 5% faster but 60% safety drop -> cliff_ratio = 0.6/0.05 = 12
        latencies = {"baseline": 500.0, "int4": 475.0}
        cliffs = detect_cliff_edges([dr], latencies)
        assert len(cliffs) >= 1
        assert cliffs[0].cliff_ratio > 3.0

    def test_failure_class_summary(self):
        grades = [
            _make_grade("s1", "x", 0.9, failure_class=FailureClass.NONE),
            _make_grade("s2", "x", 0.1, failure_class=FailureClass.A),
            _make_grade("s3", "x", 0.4, failure_class=FailureClass.B),
        ]
        summary = summarize_failure_classes(grades)
        assert summary["none"] == 1
        assert summary["A"] == 1
        assert summary["B"] == 1
