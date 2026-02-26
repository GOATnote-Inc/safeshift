"""End-to-end smoke test with mock executor."""

import json

import pytest

from safeshift.executors.mock import MockExecutor
from safeshift.grading.rubric import RubricGrader
from safeshift.scenario import load_scenarios_from_dir


class TestSmoke:
    @pytest.mark.asyncio
    async def test_single_scenario_pipeline(self, sample_scenario):
        """Full pipeline: execute -> grade -> verify."""
        executor = MockExecutor()
        grader = RubricGrader(pattern_only=True)

        result = await executor.execute(
            messages=sample_scenario.messages,
            model="mock-model",
            optimization="baseline",
        )
        assert result.response_text
        assert result.latency_ms > 0

        grade = await grader.grade(sample_scenario, result)
        assert grade.scenario_id == sample_scenario.id
        assert 0.0 <= grade.overall_score <= 1.0
        assert len(grade.dimensions) == 5

    @pytest.mark.asyncio
    async def test_multi_optimization_sweep(self, sample_scenario):
        """Test baseline vs multiple optimizations."""
        executor = MockExecutor()
        grader = RubricGrader(pattern_only=True)

        optimizations = ["baseline", "fp16", "int8", "int4"]
        results = {}

        for opt in optimizations:
            result = await executor.execute(
                messages=sample_scenario.messages,
                model="mock-model",
                optimization=opt,
            )
            grade = await grader.grade(sample_scenario, result)
            results[opt] = {
                "latency": result.latency_ms,
                "safety": grade.safety_score,
                "overall": grade.overall_score,
                "failure_class": grade.failure_class.value,
            }

        # Baseline should exist
        assert "baseline" in results
        # All results should have valid scores
        for opt, data in results.items():
            assert 0.0 <= data["safety"] <= 1.0
            assert data["latency"] > 0

    @pytest.mark.asyncio
    async def test_grade_result_serialization(self, sample_scenario):
        """Test that grade results can be serialized to JSONL."""
        executor = MockExecutor()
        grader = RubricGrader(pattern_only=True)

        result = await executor.execute(
            messages=sample_scenario.messages,
            model="mock-model",
        )
        grade = await grader.grade(sample_scenario, result)

        # Serialize and deserialize
        json_str = json.dumps(grade.to_dict())
        from safeshift.grader import GradeResult

        restored = GradeResult.from_dict(json.loads(json_str))
        assert restored.scenario_id == grade.scenario_id
        assert restored.overall_score == grade.overall_score

    @pytest.mark.asyncio
    async def test_clinical_scenarios_load_and_grade(self, scenarios_dir):
        """Load all clinical scenarios and grade with mock."""
        clinical_dir = scenarios_dir / "clinical"
        if not clinical_dir.exists():
            pytest.skip("Clinical scenarios not found")

        scenarios = load_scenarios_from_dir(clinical_dir)
        assert len(scenarios) >= 1

        executor = MockExecutor()
        grader = RubricGrader(pattern_only=True)

        for scn in scenarios[:3]:  # Test first 3 for speed
            result = await executor.execute(
                messages=scn.messages,
                model="mock-model",
                optimization="baseline",
            )
            grade = await grader.grade(scn, result)
            assert grade.scenario_id == scn.id
            assert len(grade.dimensions) == 5
