"""Base executor with timing instrumentation."""

from __future__ import annotations

import time
from typing import Any


class TimingMixin:
    """Mixin for precise latency measurement."""

    def _start_timer(self) -> dict[str, Any]:
        return {"start_ns": time.perf_counter_ns()}

    def _stop_timer(self, timer: dict[str, Any]) -> float:
        """Returns elapsed milliseconds."""
        elapsed_ns = time.perf_counter_ns() - timer["start_ns"]
        return elapsed_ns / 1_000_000
