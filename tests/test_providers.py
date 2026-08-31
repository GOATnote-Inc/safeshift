"""Tests for judge provider construction and vendor call shapes."""

import pytest

from safeshift.providers import (
    AnthropicJudgeProvider,
    MissingJudgeKeyError,
    OpenAIJudgeProvider,
    _anthropic_omits_temperature,
    _is_unsupported_temperature_error,
    build_judge_provider,
)


class TestBuildJudgeProvider:
    def test_claude_routes_to_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        provider = build_judge_provider("claude-opus-4-6")
        assert isinstance(provider, AnthropicJudgeProvider)

    def test_gpt_routes_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = build_judge_provider("gpt-5.5")
        assert isinstance(provider, OpenAIJudgeProvider)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="No judge provider"):
            build_judge_provider("nemotron30b")

    def test_missing_openai_key_fails_closed(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(MissingJudgeKeyError, match="OPENAI_API_KEY"):
            build_judge_provider("gpt-5.5")

    def test_missing_anthropic_key_fails_closed(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingJudgeKeyError, match="ANTHROPIC_API_KEY"):
            build_judge_provider("claude-opus-4-6")


class TestAnthropicTemperatureGuard:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5",
        ],
    )
    def test_new_models_omit_temperature(self, model):
        assert _anthropic_omits_temperature(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
        ],
    )
    def test_old_models_keep_temperature(self, model):
        assert _anthropic_omits_temperature(model) is False

    def test_non_claude_models_keep_temperature(self):
        assert _anthropic_omits_temperature("gpt-5.5") is False


class TestUnsupportedTemperatureDetection:
    def test_matches_openai_400_message(self):
        exc = Exception(
            "Error code: 400 - 'unsupported_value': 'temperature' does not "
            "support 0.0 with this model. Only the default (1) value is supported."
        )
        assert _is_unsupported_temperature_error(exc) is True

    def test_ignores_unrelated_errors(self):
        assert _is_unsupported_temperature_error(Exception("rate limit exceeded")) is False


class _RecordingMessages:
    def __init__(self, response):
        self.calls = []
        self._response = response

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeAnthropicResponse:
    def __init__(self, text):
        self.content = [type("Block", (), {"text": text})()]


class TestAnthropicJudgeProviderCallShape:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        return AnthropicJudgeProvider()

    async def _chat(self, provider, model):
        messages_api = _RecordingMessages(_FakeAnthropicResponse("{}"))
        provider._client = type("Client", (), {"messages": messages_api})()
        text = await provider.chat(
            messages=[
                {"role": "system", "content": "judge"},
                {"role": "user", "content": "grade this"},
            ],
            model=model,
        )
        assert text == "{}"
        assert len(messages_api.calls) == 1
        return messages_api.calls[0]

    @pytest.mark.asyncio
    async def test_omits_temperature_for_opus_4_8(self, provider):
        kwargs = await self._chat(provider, "claude-opus-4-8")
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_sends_temperature_for_sonnet_4_6(self, provider):
        kwargs = await self._chat(provider, "claude-sonnet-4-6")
        assert kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_system_message_separated(self, provider):
        kwargs = await self._chat(provider, "claude-sonnet-4-6")
        assert kwargs["system"] == "judge"
        assert all(m["role"] != "system" for m in kwargs["messages"])


class _FakeOpenAIChoice:
    def __init__(self, text):
        self.message = type("Message", (), {"content": text})()


class _FakeOpenAIResponse:
    def __init__(self, text):
        self.choices = [_FakeOpenAIChoice(text)]


class _TemperatureRejectingCompletions:
    """Rejects any call carrying temperature, mimicking reasoning models."""

    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "temperature" in kwargs:
            raise Exception(
                "Error code: 400 - 'unsupported_value': 'temperature' does not "
                "support 0.0 with this model."
            )
        return _FakeOpenAIResponse("{}")


class TestOpenAIJudgeProviderSelfHeal:
    @pytest.mark.asyncio
    async def test_retries_without_temperature_on_rejection(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        provider = OpenAIJudgeProvider()
        OpenAIJudgeProvider._models_reject_temperature.discard("test-reasoning-model")
        completions = _TemperatureRejectingCompletions()
        provider._client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": completions})()},
        )()

        text = await provider.chat(
            messages=[{"role": "user", "content": "grade this"}],
            model="test-reasoning-model",
        )
        assert text == "{}"
        assert len(completions.calls) == 2
        assert "temperature" in completions.calls[0]
        assert "temperature" not in completions.calls[1]
        # Model is remembered so the next call skips the failing attempt
        assert "test-reasoning-model" in OpenAIJudgeProvider._models_reject_temperature
        OpenAIJudgeProvider._models_reject_temperature.discard("test-reasoning-model")
