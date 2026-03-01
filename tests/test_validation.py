"""Tests for schema validation in scenario and config loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from safeshift.config import load_matrix_config
from safeshift.scenario import load_scenario


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper to write a YAML file and return its path."""

    def _write(filename: str, content: dict | str) -> Path:
        p = tmp_path / filename
        if isinstance(content, str):
            p.write_text(content)
        else:
            p.write_text(yaml.dump(content, default_flow_style=False))
        return p

    return _write


def _valid_scenario_data() -> dict:
    """Return a minimal valid scenario dict."""
    return {
        "id": "SCN-TEST-001",
        "name": "Test Scenario",
        "domain": "clinical",
        "description": "A test scenario",
        "expected_action": "Do the right thing",
        "messages": [
            {"role": "system", "content": "You are a test AI."},
            {"role": "user", "content": "Help me."},
        ],
        "latency_budget": {
            "target_ms": 500,
            "acceptable_ms": 1000,
            "critical_ms": 2000,
        },
        "safety_invariants": [
            {
                "name": "test_inv",
                "description": "Test invariant",
                "check_type": "contains",
                "pattern": "emergency",
            }
        ],
        "constraint": {
            "description": "Time critical",
            "time_pressure": "immediate",
            "consequence_of_delay": "Bad things",
        },
    }


def _valid_matrix_data() -> dict:
    """Return a minimal valid matrix config dict."""
    return {
        "name": "test-matrix",
        "scenario_paths": ["scenarios/clinical"],
        "optimization_paths": ["optimizations/baseline.yaml"],
    }


class TestScenarioValidation:
    def test_load_valid_scenario(self, tmp_yaml):
        path = tmp_yaml("valid.yaml", _valid_scenario_data())
        scn = load_scenario(path)
        assert scn.id == "SCN-TEST-001"
        assert scn.domain == "clinical"

    def test_missing_id_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["id"]
        path = tmp_yaml("no_id.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'id'"):
            load_scenario(path)

    def test_missing_name_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["name"]
        path = tmp_yaml("no_name.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'name'"):
            load_scenario(path)

    def test_missing_messages_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["messages"]
        path = tmp_yaml("no_msg.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'messages'"):
            load_scenario(path)

    def test_empty_messages_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["messages"] = []
        path = tmp_yaml("empty_msg.yaml", data)
        with pytest.raises(ValueError, match="non-empty list"):
            load_scenario(path)

    def test_messages_missing_role_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["messages"] = [{"content": "hello"}]
        path = tmp_yaml("bad_msg.yaml", data)
        with pytest.raises(ValueError, match="'role' and 'content'"):
            load_scenario(path)

    def test_invalid_domain_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["domain"] = "aerospace"
        path = tmp_yaml("bad_domain.yaml", data)
        with pytest.raises(ValueError, match="Invalid domain 'aerospace'"):
            load_scenario(path)

    def test_invalid_check_type_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["safety_invariants"][0]["check_type"] = "fuzzy_match"
        path = tmp_yaml("bad_check.yaml", data)
        with pytest.raises(ValueError, match="Invalid check_type 'fuzzy_match'"):
            load_scenario(path)

    def test_invalid_time_pressure_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["constraint"]["time_pressure"] = "whenever"
        path = tmp_yaml("bad_tp.yaml", data)
        with pytest.raises(ValueError, match="Invalid time_pressure 'whenever'"):
            load_scenario(path)

    def test_missing_latency_budget_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["latency_budget"]
        path = tmp_yaml("no_lb.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'latency_budget'"):
            load_scenario(path)

    def test_missing_constraint_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["constraint"]
        path = tmp_yaml("no_constraint.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'constraint'"):
            load_scenario(path)

    def test_missing_expected_action_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["expected_action"]
        path = tmp_yaml("no_ea.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'expected_action'"):
            load_scenario(path)

    def test_non_numeric_severity_raises_valueerror(self, tmp_yaml):
        data = _valid_scenario_data()
        data["severity"] = "high"
        path = tmp_yaml("bad_severity.yaml", data)
        with pytest.raises(ValueError, match="severity must be numeric"):
            load_scenario(path)

    def test_malformed_yaml_raises_valueerror(self, tmp_yaml):
        path = tmp_yaml("bad.yaml", "{{{{not yaml")
        with pytest.raises(ValueError, match="Failed to load scenario"):
            load_scenario(path)

    def test_error_includes_file_path(self, tmp_yaml):
        data = _valid_scenario_data()
        del data["id"]
        path = tmp_yaml("context.yaml", data)
        with pytest.raises(ValueError) as exc_info:
            load_scenario(path)
        assert str(path) in str(exc_info.value)

    def test_valid_domains_accepted(self, tmp_yaml):
        for domain in ("clinical", "robotic", "clinical_defer", "robotic_defer"):
            data = _valid_scenario_data()
            data["domain"] = domain
            data["id"] = f"SCN-{domain}"
            path = tmp_yaml(f"{domain}.yaml", data)
            scn = load_scenario(path)
            assert scn.domain == domain

    def test_optional_fields_have_defaults(self, tmp_yaml):
        data = _valid_scenario_data()
        # Remove all optional fields
        data.pop("safety_invariants", None)
        path = tmp_yaml("minimal.yaml", data)
        scn = load_scenario(path)
        assert scn.severity == 1.0
        assert scn.tags == []
        assert scn.metadata == {}
        assert scn.rubric == "rubric_clinical"
        assert scn.safety_invariants == []


class TestMatrixConfigValidation:
    def test_load_valid_config(self, tmp_yaml):
        path = tmp_yaml("valid.yaml", _valid_matrix_data())
        cfg = load_matrix_config(path)
        assert cfg.name == "test-matrix"

    def test_missing_name_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        del data["name"]
        path = tmp_yaml("no_name.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'name'"):
            load_matrix_config(path)

    def test_missing_scenario_paths_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        del data["scenario_paths"]
        path = tmp_yaml("no_sp.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'scenario_paths'"):
            load_matrix_config(path)

    def test_missing_optimization_paths_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        del data["optimization_paths"]
        path = tmp_yaml("no_op.yaml", data)
        with pytest.raises(ValueError, match="Missing required field 'optimization_paths'"):
            load_matrix_config(path)

    def test_non_numeric_temperature_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        data["temperature"] = "warm"
        path = tmp_yaml("bad_temp.yaml", data)
        with pytest.raises(ValueError, match="temperature must be numeric"):
            load_matrix_config(path)

    def test_non_int_seed_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        data["seed"] = 3.14
        path = tmp_yaml("bad_seed.yaml", data)
        with pytest.raises(ValueError, match="seed must be an integer"):
            load_matrix_config(path)

    def test_non_int_n_trials_raises_valueerror(self, tmp_yaml):
        data = _valid_matrix_data()
        data["n_trials"] = 2.5
        path = tmp_yaml("bad_nt.yaml", data)
        with pytest.raises(ValueError, match="n_trials must be an integer"):
            load_matrix_config(path)

    def test_optional_fields_have_defaults(self, tmp_yaml):
        path = tmp_yaml("minimal.yaml", _valid_matrix_data())
        cfg = load_matrix_config(path)
        assert cfg.temperature == 0.0
        assert cfg.seed == 42
        assert cfg.n_trials == 1
        assert cfg.executor == "mock"
        assert cfg.model == "mock-model"

    def test_malformed_yaml_raises_valueerror(self, tmp_yaml):
        path = tmp_yaml("bad.yaml", "{{{{not yaml")
        with pytest.raises(ValueError, match="Failed to load matrix config"):
            load_matrix_config(path)

    def test_error_includes_file_path(self, tmp_yaml):
        data = _valid_matrix_data()
        del data["name"]
        path = tmp_yaml("ctx.yaml", data)
        with pytest.raises(ValueError) as exc_info:
            load_matrix_config(path)
        assert str(path) in str(exc_info.value)
