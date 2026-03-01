"""Cloud API executor (OpenAI / Anthropic)."""

from __future__ import annotations

import logging
import os

from safeshift.executor import Executor, ExecutorResult
from safeshift.executors.base import TimingMixin
from safeshift.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class APIExecutor(Executor, TimingMixin):
    """Execute inference via cloud APIs (OpenAI or Anthropic)."""

    def __init__(self, provider: str = "openai", **kwargs):
        self._provider = provider
        self._client = None
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return f"api:{self._provider}"

    def _get_openai_client(self):
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            )
        return self._client

    def _get_anthropic_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
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
        timer = self._start_timer()

        if self._provider == "anthropic":
            result = await self._execute_anthropic(messages, model, temperature, max_tokens)
        else:
            result = await self._execute_openai(messages, model, temperature, seed, max_tokens)

        latency_ms = self._stop_timer(timer)

        return ExecutorResult(
            response_text=result["text"],
            latency_ms=round(latency_ms, 2),
            total_tokens=result.get("total_tokens"),
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
            model=model,
            optimization=optimization,
            executor_type=self.name,
        )

    async def _execute_openai(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> dict:
        client = self._get_openai_client()

        response = await retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_completion_tokens=max_tokens,
            )
        )

        if not response.choices:
            raise ValueError("Empty response from OpenAI: no choices returned")

        choice = response.choices[0]
        usage = response.usage
        return {
            "text": choice.message.content or "",
            "total_tokens": usage.total_tokens if usage else None,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
        }

    async def _execute_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        client = self._get_anthropic_client()

        # Separate system message
        system = None
        api_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                api_messages.append(msg)

        kwargs = {
            "model": model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system

        response = await retry_with_backoff(lambda: client.messages.create(**kwargs))

        if not response.content:
            raise ValueError("Empty response from Anthropic: no content blocks returned")

        text = response.content[0].text
        return {
            "text": text,
            "total_tokens": (response.usage.input_tokens + response.usage.output_tokens)
            if response.usage
            else None,
            "prompt_tokens": response.usage.input_tokens if response.usage else None,
            "completion_tokens": response.usage.output_tokens if response.usage else None,
        }

    async def health_check(self) -> bool:
        if self._provider == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY"))
        return bool(os.environ.get("OPENAI_API_KEY"))

    async def close(self) -> None:
        self._client = None
