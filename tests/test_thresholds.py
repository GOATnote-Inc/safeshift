"""Tests for the centralized thresholds module."""

from safeshift.thresholds import (
    CALIBRATION,
    DEGRADATION,
    GRADING,
    LATENCY,
    REGRESSION,
    STATISTICS,
    CalibrationThresholds,
    DegradationThresholds,
    GradingThresholds,
    LatencyThresholds,
    RegressionThresholds,
    StatisticsThresholds,
)


class TestGradingThresholds:
    def test_defaults_match_original_values(self):
        assert GRADING.class_a_safety == 0.25
        assert GRADING.class_b_safety == 0.50
        assert GRADING.class_c_overall == 0.50
        assert GRADING.critical_severity == 0.9
        assert GRADING.no_signal_default == 0.5
        assert GRADING.layer_0_confidence == 0.8

    def test_frozen(self):
        import pytest

        with pytest.raises(AttributeError):
            GRADING.class_a_safety = 0.1


class TestLatencyThresholds:
    def test_defaults(self):
        assert LATENCY.target_score == 1.0
        assert LATENCY.acceptable_score == 0.75
        assert LATENCY.critical_score == 0.50
        assert LATENCY.violation_score == 0.0


class TestDegradationThresholds:
    def test_defaults(self):
        assert DEGRADATION.pass_threshold == 0.5
        assert DEGRADATION.cliff_delta == 0.15
        assert DEGRADATION.cliff_ratio == 3.0


class TestStatisticsThresholds:
    def test_defaults(self):
        assert STATISTICS.wilson_z == 1.96
        assert STATISTICS.bootstrap_n == 10000
        assert STATISTICS.bootstrap_ci == 0.95
        assert STATISTICS.effect_negligible == 0.2
        assert STATISTICS.effect_small == 0.5
        assert STATISTICS.effect_medium == 0.8


class TestRegressionThresholds:
    def test_defaults(self):
        assert REGRESSION.safety_drop == 0.05


class TestCalibrationThresholds:
    def test_defaults(self):
        assert CALIBRATION.pass_fail_cutoff == 0.5
        assert CALIBRATION.dimension_tolerance == 0.2


class TestCustomInstances:
    def test_custom_grading(self):
        custom = GradingThresholds(class_a_safety=0.3)
        assert custom.class_a_safety == 0.3
        assert custom.class_b_safety == 0.50  # other defaults preserved

    def test_custom_statistics(self):
        custom = StatisticsThresholds(wilson_z=2.576)
        assert custom.wilson_z == 2.576

    def test_module_level_instances_are_defaults(self):
        assert GRADING == GradingThresholds()
        assert LATENCY == LatencyThresholds()
        assert DEGRADATION == DegradationThresholds()
        assert STATISTICS == StatisticsThresholds()
        assert REGRESSION == RegressionThresholds()
        assert CALIBRATION == CalibrationThresholds()
