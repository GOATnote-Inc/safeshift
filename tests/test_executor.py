"""Tests for executor backends."""

import pytest

from safeshift.executor import ExecutorResult
from safeshift.executors import get_executor
from safeshift.executors.mock import MockExecutor


class TestExecutorResult:
    def test_to_dict_roundtrip(self):
        result = ExecutorResult(
            response_text="test response",
            latency_ms=100.5,
            model="test-model",
            optimization="fp16",
            scenario_id="SCN-001",
        )
        d = result.to_dict()
        restored = ExecutorResult.from_dict(d)
        assert restored.response_text == result.response_text
        assert restored.latency_ms == result.latency_ms
        assert restored.model == result.model
        assert restored.optimization == result.optimization

    def test_frozen(self):
        result = ExecutorResult(response_text="test", latency_ms=100.0)
        with pytest.raises(AttributeError):
            result.response_text = "new"


class TestMockExecutor:
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        executor = MockExecutor()
        result = await executor.execute(
            messages=[{"role": "user", "content": "Patient with chest pain"}],
            model="mock-model",
        )
        assert result.response_text
        assert result.latency_ms > 0
        assert result.executor_type == "mock"

    @pytest.mark.asyncio
    async def test_clinical_response(self):
        executor = MockExecutor()
        result = await executor.execute(
            messages=[{"role": "user", "content": "Patient with clinical emergency"}],
            model="mock-model",
        )
        text = result.response_text.lower()
        assert "patient" in text or "emergency" in text

    @pytest.mark.asyncio
    async def test_robotic_response(self):
        executor = MockExecutor()
        result = await executor.execute(
            messages=[{"role": "user", "content": "Robot arm in workspace danger"}],
            model="mock-model",
        )
        assert result.response_text

    @pytest.mark.asyncio
    async def test_deterministic_same_seed(self):
        executor = MockExecutor()
        msg = [{"role": "user", "content": "Patient emergency triage"}]
        r1 = await executor.execute(msg, "mock-model", seed=42)
        r2 = await executor.execute(msg, "mock-model", seed=42)
        assert r1.response_text == r2.response_text

    @pytest.mark.asyncio
    async def test_different_optimization_different_latency(self):
        executor = MockExecutor()
        msg = [{"role": "user", "content": "Clinical triage needed"}]
        r_baseline = await executor.execute(msg, "mock-model", optimization="baseline")
        r_int4 = await executor.execute(msg, "mock-model", optimization="int4")
        # INT4 should generally be faster (lower mean latency)
        # Note: stochastic, but with fixed seeds this is deterministic
        assert r_baseline.latency_ms != r_int4.latency_ms

    @pytest.mark.asyncio
    async def test_supports_optimization(self):
        executor = MockExecutor()
        assert executor.supports_optimization is True

    @pytest.mark.asyncio
    async def test_name(self):
        executor = MockExecutor()
        assert executor.name == "mock"

    @pytest.mark.asyncio
    async def test_health_check(self):
        executor = MockExecutor()
        assert await executor.health_check() is True


class TestGetExecutor:
    def test_get_mock(self):
        exc = get_executor("mock")
        assert isinstance(exc, MockExecutor)

    def test_unknown_executor(self):
        with pytest.raises(ValueError, match="Unknown executor"):
            get_executor("nonexistent")
