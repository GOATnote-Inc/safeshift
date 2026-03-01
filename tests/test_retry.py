"""Tests for retry logic and circuit breaker."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from safeshift.retry import (
    CircuitBreaker,
    CircuitOpenError,
    _is_retryable_status,
    reset_circuit_breaker,
    retry_with_backoff,
)


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=3)
        assert not cb.is_open

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open

    def test_success_resets_counter(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open

    def test_reset(self):
        cb = CircuitBreaker(threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert not cb.is_open

    def test_default_threshold(self):
        cb = CircuitBreaker()
        for _ in range(4):
            cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open


class TestRetryWithBackoff:
    @pytest.fixture(autouse=True)
    def _reset_breaker(self):
        """Reset the module-level circuit breaker before each test."""
        reset_circuit_breaker()
        yield
        reset_circuit_breaker()

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        mock = AsyncMock(return_value="ok")
        result = await retry_with_backoff(mock)
        assert result == "ok"
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        mock = AsyncMock(side_effect=[TimeoutError("timeout"), "ok"])
        result = await retry_with_backoff(mock, max_retries=2)
        assert result == "ok"
        assert mock.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        mock = AsyncMock(side_effect=TimeoutError("always fails"))
        with pytest.raises(TimeoutError):
            await retry_with_backoff(mock, max_retries=2)
        assert mock.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_propagates_immediately(self):
        mock = AsyncMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            await retry_with_backoff(mock, max_retries=3)
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_calls(self):
        # Manually open the circuit breaker
        from safeshift.retry import _circuit_breaker

        for _ in range(5):
            _circuit_breaker.record_failure()

        mock = AsyncMock(return_value="ok")
        with pytest.raises(CircuitOpenError):
            await retry_with_backoff(mock)
        assert mock.call_count == 0


class TestIsRetryableStatus:
    def test_non_api_error_not_retryable(self):
        assert not _is_retryable_status(ValueError("nope"))

    def test_timeout_error_not_status(self):
        assert not _is_retryable_status(TimeoutError("timeout"))
