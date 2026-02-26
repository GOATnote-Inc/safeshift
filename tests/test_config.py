"""Tests for configuration loading."""

import pytest

from safeshift.config import MatrixConfig, RunConfig, load_matrix_config


class TestRunConfig:
    def test_valid(self):
        config = RunConfig()
        assert config.validate() == []

    def test_invalid_temperature(self):
        config = RunConfig(temperature=0.7)
        errors = config.validate()
        assert any("temperature" in e for e in errors)

    def test_frozen(self):
        config = RunConfig()
        with pytest.raises(AttributeError):
            config.seed = 123


class TestMatrixConfig:
    def test_valid(self):
        config = MatrixConfig(
            name="test",
            description="test matrix",
            scenario_paths=["scenarios/"],
            optimization_paths=["optimizations/quantization.yaml"],
        )
        assert config.validate() == []

    def test_missing_scenarios(self):
        config = MatrixConfig(
            name="test",
            description="",
            scenario_paths=[],
            optimization_paths=["optimizations/quantization.yaml"],
        )
        errors = config.validate()
        assert any("scenario" in e for e in errors)

    def test_missing_optimizations(self):
        config = MatrixConfig(
            name="test",
            description="",
            scenario_paths=["scenarios/"],
            optimization_paths=[],
        )
        errors = config.validate()
        assert any("optimization" in e for e in errors)


class TestLoadMatrixConfig:
    def test_load_quick_matrix(self, configs_dir):
        path = configs_dir / "matrices" / "quick_matrix.yaml"
        if not path.exists():
            pytest.skip("Matrix config not found")
        config = load_matrix_config(path)
        assert config.name == "quick_matrix"
        assert len(config.scenario_paths) >= 1
        assert config.temperature == 0.0
        assert config.seed == 42

    def test_load_default_matrix(self, configs_dir):
        path = configs_dir / "matrices" / "default_matrix.yaml"
        if not path.exists():
            pytest.skip("Matrix config not found")
        config = load_matrix_config(path)
        assert config.name == "default_matrix"
