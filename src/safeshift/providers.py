"""Judge provider construction — the seam between the CLI and vendor SDKs.

A judge provider is any object exposing::

    async def chat(messages, model, temperature=0.0, seed=42) -> str

``build_judge_provider`` maps a judge model ID to a concrete provider and
fails closed: a missing API key raises ``MissingJudgeKeyError`` before any
evaluation work starts, so a multi-hour run can never silently fall back to
pattern-only grades.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Default judge model. Override with SAFESHIFT_JUDGE_MODEL or --judge-model.
DEFAULT_JUDGE_MODEL = os.environ.get("SAFESHIFT_JUDGE_MODEL", "gpt-5.5")


class MissingJudgeKeyError(RuntimeError):
    """Raised when the judge provider's API key is not set."""


def _anthropic_omits_temperature(model: str) -> bool:
    """True when the Anthropic API rejects an explicit ``temperature`` (400).

    Opus 4.7+ and all major-version-5 families (opus-5, sonnet-5, fable-5)
    deprecated the parameter. Older models (opus <= 4-6, sonnet-4-6,
    haiku-4-5) still accept it and MUST keep receiving it — omitting it there
    would silently fall back to the API default and break temp=0 replay.
    """
    m = re.search(r"claude-([a-z]+)-(\d+)(?:[.-](\d+))?", model.lower())
    if not m:
        return False
    family, major = m.group(1), int(m.group(2))
    minor = int(m.group(3)) if m.group(3) is not None else 0
    if major >= 5:
        return True
    return family == "opus" and (major, minor) >= (4, 7)


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    """True for the OpenAI 'model only supports default temperature' 400.

    Emitted by reasoning models (e.g. gpt-5.5, o-series) that reject an
    explicit temperature=0. Detected by message so no SDK error class import
    is needed.
    """
    s = str(exc).lower()
    return "temperature" in s and (
        "does not support" in s or "unsupported_value" in s or "only the default" in s
    )


class OpenAIJudgeProvider:
    """Judge provider backed by the OpenAI chat completions API."""

    # Models discovered at runtime to reject an explicit temperature.
    # Process-shared so only the first call per model pays the retry.
    _models_reject_temperature: set[str] = set()

    def __init__(self) -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise MissingJudgeKeyError(
                "OPENAI_API_KEY is not set but the judge model requires it. "
                "Export the key, pick a different --judge-model, or pass "
                "--pattern-only to explicitly opt out of LLM judging."
            )
        import openai

        self._client = openai.AsyncOpenAI()

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        from safeshift.retry import retry_with_backoff

        kwargs: dict = {"model": model, "messages": messages, "seed": seed}
        if model not in OpenAIJudgeProvider._models_reject_temperature:
            kwargs["temperature"] = temperature

        try:
            response = await retry_with_backoff(
                lambda: self._client.chat.completions.create(**kwargs)
            )
        except Exception as e:
            # Self-heal reasoning models that reject an explicit temperature.
            if "temperature" not in kwargs or not _is_unsupported_temperature_error(e):
                raise
            logger.warning(
                "%s rejects explicit temperature; retrying without it. "
                "Judge determinism relies on seed only for this model.",
                model,
            )
            OpenAIJudgeProvider._models_reject_temperature.add(model)
            kwargs.pop("temperature")
            response = await retry_with_backoff(
                lambda: self._client.chat.completions.create(**kwargs)
            )

        if not response.choices:
            raise ValueError(f"Empty judge response from OpenAI (model={model})")
        return response.choices[0].message.content or ""


class AnthropicJudgeProvider:
    """Judge provider backed by the Anthropic messages API."""

    def __init__(self) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise MissingJudgeKeyError(
                "ANTHROPIC_API_KEY is not set but the judge model requires it. "
                "Export the key, pick a different --judge-model, or pass "
                "--pattern-only to explicitly opt out of LLM judging."
            )
        import anthropic

        self._client = anthropic.AsyncAnthropic()

    async def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        from safeshift.retry import retry_with_backoff

        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict = {"model": model, "messages": chat_messages, "max_tokens": 4096}
        if system:
            kwargs["system"] = system
        if not _anthropic_omits_temperature(model):
            kwargs["temperature"] = temperature

        response = await retry_with_backoff(lambda: self._client.messages.create(**kwargs))
        if not response.content:
            raise ValueError(f"Empty judge response from Anthropic (model={model})")
        return response.content[0].text or ""


def build_judge_provider(judge_model: str):
    """Build the judge provider for ``judge_model``.

    Fails closed: raises ``MissingJudgeKeyError`` when the key is absent and
    ``ValueError`` when no provider matches the model ID.
    """
    lowered = judge_model.lower()
    if "claude" in lowered:
        return AnthropicJudgeProvider()
    if "gpt" in lowered or lowered.startswith("o"):
        return OpenAIJudgeProvider()
    raise ValueError(
        f"No judge provider available for model '{judge_model}'. "
        "Supported judge models must contain 'claude' or 'gpt'."
    )
