"""Tests for optimization configuration."""

import pytest

from safeshift.optimizer import OptimizationAxis, OptimizationConfig, load_optimizations


class TestOptimizationAxis:
    def test_str(self):
        axis = OptimizationAxis("quantization", "fp16")
        assert str(axis) == "quantization=fp16"

    def test_frozen(self):
        axis = OptimizationAxis("quantization", "fp16")
        with pytest.raises(AttributeError):
            axis.name = "other"


class TestOptimizationConfig:
    def test_baseline_label(self):
        config = OptimizationConfig(
            id="baseline",
            name="Baseline",
            axes=[],
            is_baseline=True,
        )
        assert config.label == "baseline"

    def test_multi_axis_label(self):
        config = OptimizationConfig(
            id="combo",
            name="Combo",
            axes=[
                OptimizationAxis("quantization", "int8"),
                OptimizationAxis("batch_size", 16),
            ],
        )
        assert config.label == "quantization=int8+batch_size=16"

    def test_to_executor_params(self):
        config = OptimizationConfig(
            id="test",
            name="Test",
            axes=[
                OptimizationAxis("quantization", "fp16"),
                OptimizationAxis("batch_size", 8),
            ],
        )
        params = config.to_executor_params()
        assert params["quantization"] == "fp16"
        assert params["batch_size"] == 8


class TestLoadOptimizations:
    def test_load_quantization(self, configs_dir):
        path = configs_dir / "optimizations" / "quantization.yaml"
        if not path.exists():
            pytest.skip("Optimization config not found")
        opts = load_optimizations(path)
        assert len(opts) == 5
        baselines = [o for o in opts if o.is_baseline]
        assert len(baselines) == 1

    def test_load_batching(self, configs_dir):
        path = configs_dir / "optimizations" / "batching.yaml"
        if not path.exists():
            pytest.skip("Optimization config not found")
        opts = load_optimizations(path)
        assert len(opts) == 5

    def test_load_composite(self, configs_dir):
        path = configs_dir / "optimizations" / "composite.yaml"
        if not path.exists():
            pytest.skip("Optimization config not found")
        opts = load_optimizations(path)
        assert len(opts) >= 3
        # Composite configs should have multiple axes
        for opt in opts:
            assert len(opt.axes) >= 2
