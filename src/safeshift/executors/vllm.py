"""vLLM executor — OpenAI-compatible API backed by vLLM server."""

from __future__ import annotations

import logging
import os

from safeshift.executor import Executor, ExecutorResult
from safeshift.executors.base import TimingMixin

logger = logging.getLogger(__name__)


class VLLMExecutor(Executor, TimingMixin):
    """Execute inference via vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, **kwargs):
        self._base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self._client = None

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def supports_optimization(self) -> bool:
        return True

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    async def execute(
        self,
        messages: list[dict[str, str]],
        model: str,
        optimization: str = "baseline",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 4096,
    ) -> ExecutorResult:
        client = self._get_client()
        timer = self._start_timer()

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            seed=seed,
            max_completion_tokens=max_tokens,
        )

        latency_ms = self._stop_timer(timer)
        choice = response.choices[0]
        usage = response.usage
        completion_tokens = usage.completion_tokens if usage else None
        tokens_per_sec = None
        if completion_tokens and latency_ms > 0:
            tokens_per_sec = round(completion_tokens / (latency_ms / 1000), 1)

        return ExecutorResult(
            response_text=choice.message.content or "",
            latency_ms=round(latency_ms, 2),
            tokens_per_sec=tokens_per_sec,
            total_tokens=usage.total_tokens if usage else None,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=completion_tokens,
            model=model,
            optimization=optimization,
            executor_type="vllm",
        )

    async def health_check(self) -> bool:
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            logger.warning("vLLM health check failed at %s", self._base_url)
            return False

    async def close(self) -> None:
        self._client = None
