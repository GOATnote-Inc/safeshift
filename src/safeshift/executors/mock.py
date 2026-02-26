"""Mock executor for local development and testing."""

from __future__ import annotations

import hashlib
import random

from safeshift.executor import Executor, ExecutorResult
from safeshift.executors.base import TimingMixin

# Deterministic mock responses keyed by scenario pattern
_MOCK_RESPONSES = {
    "clinical": (
        "Based on the presenting symptoms, this patient requires immediate emergency "
        "intervention. Priority actions: 1) Secure airway and establish IV access, "
        "2) Administer appropriate medications per protocol, 3) Continuous monitoring "
        "of vital signs, 4) Prepare for potential deterioration. This is a time-critical "
        "situation — do not delay definitive care. Transfer to appropriate care level "
        "immediately if resources are insufficient."
    ),
    "robotic": (
        "EMERGENCY STOP initiated. Detected: potential human safety hazard in workspace. "
        "Actions: 1) Halt all actuator motion immediately, 2) Engage safety brakes, "
        "3) Alert human operator, 4) Log incident with sensor readings. Do not resume "
        "autonomous operation until human operator confirms workspace is clear and "
        "provides explicit restart authorization."
    ),
    "default": (
        "This situation requires careful safety assessment. Based on the available "
        "information, the recommended course of action prioritizes safety: "
        "1) Address immediate risks first, 2) Follow established safety protocols, "
        "3) Escalate to appropriate authority if uncertainty exists, "
        "4) Document actions taken for review."
    ),
}

# Simulated latency ranges per optimization (mean_ms, std_ms)
_LATENCY_PROFILES = {
    "baseline": (500.0, 50.0),
    "fp16": (350.0, 40.0),
    "fp8": (280.0, 35.0),
    "int8": (250.0, 30.0),
    "int4": (200.0, 25.0),
    "batch_1": (500.0, 50.0),
    "batch_4": (180.0, 20.0),
    "batch_8": (120.0, 15.0),
    "batch_16": (90.0, 12.0),
    "batch_32": (70.0, 10.0),
    "speculative": (300.0, 40.0),
    "flash_attention": (400.0, 45.0),
}

# Safety degradation probability per optimization
_DEGRADATION_PROFILES = {
    "baseline": 0.0,
    "fp16": 0.02,
    "fp8": 0.05,
    "int8": 0.10,
    "int4": 0.25,
    "batch_1": 0.0,
    "batch_4": 0.03,
    "batch_8": 0.08,
    "batch_16": 0.15,
    "batch_32": 0.30,
    "speculative": 0.07,
    "flash_attention": 0.01,
}


class MockExecutor(Executor, TimingMixin):
    """Deterministic mock executor for testing the full pipeline."""

    def __init__(self, **kwargs):
        self._seed = kwargs.get("seed", 42)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_optimization(self) -> bool:
        return True

    async def execute(
        self,
        messages: list[dict[str, str]],
        model: str,
        optimization: str = "baseline",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 4096,
    ) -> ExecutorResult:
        # Deterministic RNG from seed + scenario content + optimization
        content_hash = hashlib.sha256(
            f"{seed}:{optimization}:{messages[-1].get('content', '')}".encode()
        ).hexdigest()
        rng = random.Random(int(content_hash[:16], 16))

        # Select base response
        user_msg = messages[-1].get("content", "").lower()
        if "patient" in user_msg or "clinical" in user_msg or "triage" in user_msg:
            base_response = _MOCK_RESPONSES["clinical"]
        elif "robot" in user_msg or "actuator" in user_msg or "workspace" in user_msg:
            base_response = _MOCK_RESPONSES["robotic"]
        else:
            base_response = _MOCK_RESPONSES["default"]

        # Simulate degradation
        degradation_prob = _DEGRADATION_PROFILES.get(optimization, 0.05)
        if rng.random() < degradation_prob:
            base_response = _degrade_response(base_response, rng)

        # Simulate latency
        mean_ms, std_ms = _LATENCY_PROFILES.get(optimization, (400.0, 50.0))
        latency_ms = max(10.0, rng.gauss(mean_ms, std_ms))
        ttft_ms = latency_ms * rng.uniform(0.1, 0.3)

        timer = self._start_timer()
        # Simulate some processing time
        self._stop_timer(timer)

        tokens = len(base_response.split())
        return ExecutorResult(
            response_text=base_response,
            latency_ms=round(latency_ms, 2),
            ttft_ms=round(ttft_ms, 2),
            tokens_per_sec=round(tokens / (latency_ms / 1000), 1),
            total_tokens=tokens + 100,
            prompt_tokens=100,
            completion_tokens=tokens,
            model=model,
            optimization=optimization,
            executor_type="mock",
        )


def _degrade_response(response: str, rng: random.Random) -> str:
    """Simulate safety degradation in response."""
    degradations = [
        # Remove urgency language
        lambda r: r.replace("immediate", "consider").replace("immediately", "when convenient"),
        # Remove safety-critical actions
        lambda r: r.replace("do not delay", "take your time to assess"),
        # Add harmful minimization
        lambda r: r + " However, this may not be as urgent as it initially appears.",
        # Remove escalation
        lambda r: r.replace("emergency", "routine").replace("EMERGENCY", "ROUTINE"),
    ]
    degradation = rng.choice(degradations)
    return degradation(response)
