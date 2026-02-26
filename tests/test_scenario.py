"""Tests for scenario loading and validation."""

import pytest

from safeshift.scenario import (
    LatencyBudget,
    SafetyInvariant,
    load_scenario,
    load_scenarios_from_dir,
)


class TestLatencyBudget:
    def test_classify_target(self):
        lb = LatencyBudget(500, 1000, 2000)
        assert lb.classify(300) == "target"

    def test_classify_acceptable(self):
        lb = LatencyBudget(500, 1000, 2000)
        assert lb.classify(750) == "acceptable"

    def test_classify_critical(self):
        lb = LatencyBudget(500, 1000, 2000)
        assert lb.classify(1500) == "critical"

    def test_classify_violation(self):
        lb = LatencyBudget(500, 1000, 2000)
        assert lb.classify(3000) == "violation"

    def test_classify_boundary_target(self):
        lb = LatencyBudget(500, 1000, 2000)
        assert lb.classify(500) == "target"


class TestSafetyInvariant:
    def test_regex_match(self):
        inv = SafetyInvariant("test", "test", "regex", r"\bemergency\b")
        assert inv.check_deterministic("This is an emergency situation") is True

    def test_regex_no_match(self):
        inv = SafetyInvariant("test", "test", "regex", r"\bemergency\b")
        assert inv.check_deterministic("This is a routine situation") is False

    def test_contains_match(self):
        inv = SafetyInvariant("test", "test", "contains", "emergency")
        assert inv.check_deterministic("Call emergency services") is True

    def test_not_contains_pass(self):
        inv = SafetyInvariant("test", "test", "not_contains", "not urgent")
        assert inv.check_deterministic("This is very urgent") is True

    def test_not_contains_fail(self):
        inv = SafetyInvariant("test", "test", "not_contains", "not urgent")
        assert inv.check_deterministic("This is not urgent at all") is False

    def test_judge_returns_none(self):
        inv = SafetyInvariant("test", "test", "judge", judge_criterion="Check safety")
        assert inv.check_deterministic("any text") is None


class TestScenarioLoading:
    def test_load_clinical_scenario(self, scenarios_dir):
        path = scenarios_dir / "clinical" / "stemi.yaml"
        if not path.exists():
            pytest.skip("Scenario file not found")
        scenario = load_scenario(path)
        assert scenario.id == "SCN-C-001"
        assert scenario.domain == "clinical"
        assert len(scenario.messages) >= 2
        assert len(scenario.safety_invariants) >= 1
        assert scenario.severity == 1.0

    def test_load_robotic_scenario(self, scenarios_dir):
        path = scenarios_dir / "robotic" / "human_proximity.yaml"
        if not path.exists():
            pytest.skip("Scenario file not found")
        scenario = load_scenario(path)
        assert scenario.id == "SCN-R-001"
        assert scenario.domain == "robotic"

    def test_load_scenarios_from_dir(self, scenarios_dir):
        clinical_dir = scenarios_dir / "clinical"
        if not clinical_dir.exists():
            pytest.skip("Scenarios directory not found")
        scenarios = load_scenarios_from_dir(clinical_dir)
        assert len(scenarios) >= 1
        assert all(s.domain == "clinical" for s in scenarios)

    def test_scenario_frozen(self, sample_scenario):
        with pytest.raises(AttributeError):
            sample_scenario.id = "new-id"
