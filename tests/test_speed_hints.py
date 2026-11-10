"""Per-turn `speed` hint: config, resolution, and provider application.

The hint is client-sent per user_message, resolved against the agent's
provider-type baseline + per-agent overrides, and reaches the adapter as
a `speed_options` dict. Here we exercise every layer:

- Resolution semantics (null disables, missing uses baseline, dict overrides).
- OpenRouter adapter translates `provider_sort` into the OpenAI SDK's
  `extra_body={"provider": {"sort": ...}}`.
- Non-OpenRouter providers silently ignore the hint.
"""

import pytest

from ark.config import (
    AgentConfig,
    Config,
    ConfigError,
    ProviderConfig,
    ServerConfig,
    _from_dict,
)
from ark.provider import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    OpenRouterProvider,
)
from ark.runtime import effective_speed_hints, resolve_speed_options


# ---------------------------------------------------------------------------
# Provider-type baselines
# ---------------------------------------------------------------------------


def test_openrouter_has_fast_latency_baseline():
    """OpenRouter ships with `fast -> provider_sort: latency` so new users
    get the hint for free without editing config."""
    assert OpenRouterProvider.DEFAULT_SPEED_HINTS == {
        "fast": {"provider_sort": "latency"}
    }


def test_direct_providers_have_no_baseline():
    """Anthropic/OpenAI/Google direct have no latency sort — the hint is
    silently a no-op there."""
    assert AnthropicProvider.DEFAULT_SPEED_HINTS == {}
    assert OpenAIProvider.DEFAULT_SPEED_HINTS == {}
    assert GoogleProvider.DEFAULT_SPEED_HINTS == {}


# ---------------------------------------------------------------------------
# effective_speed_hints: merging baseline + agent overrides
# ---------------------------------------------------------------------------


def _agent(**kw) -> AgentConfig:
    base = dict(name="scribe", provider="p", model="m", workspace="/tmp/ws")
    base.update(kw)
    return AgentConfig(**base)


def test_effective_missing_uses_baseline():
    """Agent doesn't mention `fast` → OpenRouter's baseline applies."""
    a = _agent()
    assert effective_speed_hints(a, "openrouter") == {
        "fast": {"provider_sort": "latency"}
    }


def test_effective_override_replaces_baseline():
    """Agent-level override wins over the provider-type default."""
    a = _agent(speed_hints={"fast": {"provider_sort": "throughput"}})
    assert effective_speed_hints(a, "openrouter") == {
        "fast": {"provider_sort": "throughput"}
    }


def test_effective_null_disables_baseline():
    """`speed_hints.fast = null` means "ignore the provider default on this
    agent" — different from "missing", which keeps the default."""
    a = _agent(speed_hints={"fast": None})
    assert effective_speed_hints(a, "openrouter") == {"fast": None}


def test_effective_anthropic_stays_empty():
    """Even with a `fast` override, Anthropic has no builtin and the
    agent-added entry still rides into the merged map (provider ignores it)."""
    a = _agent(speed_hints={"fast": {"provider_sort": "latency"}})
    assert effective_speed_hints(a, "anthropic") == {
        "fast": {"provider_sort": "latency"}
    }


# ---------------------------------------------------------------------------
# resolve_speed_options: what the adapter actually sees per turn
# ---------------------------------------------------------------------------


def test_resolve_no_hint_returns_none():
    """No hint sent by the client → no options; adapter adds nothing."""
    a = _agent()
    assert resolve_speed_options(a, "openrouter", None) is None


def test_resolve_fast_on_openrouter_returns_latency():
    a = _agent()
    assert resolve_speed_options(a, "openrouter", "fast") == {
        "provider_sort": "latency"
    }


def test_resolve_fast_on_anthropic_returns_none():
    """No baseline + no agent override → nothing to apply → None."""
    a = _agent()
    assert resolve_speed_options(a, "anthropic", "fast") is None


def test_resolve_fast_disabled_returns_none():
    """Explicit null on the agent disables the hint even on OpenRouter."""
    a = _agent(speed_hints={"fast": None})
    assert resolve_speed_options(a, "openrouter", "fast") is None


def test_resolve_unknown_hint_returns_none():
    """An unknown hint value is a silent no-op — forward-compatible with
    future hints without breaking agents that don't know them."""
    a = _agent()
    assert resolve_speed_options(a, "openrouter", "turbo") is None


# ---------------------------------------------------------------------------
# OpenRouter adapter: options → extra_body shape
# ---------------------------------------------------------------------------


def test_openrouter_build_extra_body_latency():
    p = OpenRouterProvider(api_key="or-test")
    assert p._build_extra_body({"provider_sort": "latency"}) == {
        "provider": {"sort": "latency"}
    }


def test_openrouter_build_extra_body_throughput():
    """Same mechanism also supports `throughput` and `price`, so agents can
    configure an alternative mapping without a code change."""
    p = OpenRouterProvider(api_key="or-test")
    assert p._build_extra_body({"provider_sort": "throughput"}) == {
        "provider": {"sort": "throughput"}
    }


def test_openrouter_build_extra_body_rejects_unknown_sort():
    """Defensive: an unknown `provider_sort` is dropped rather than
    forwarded — we won't 400 the OpenRouter call on a config typo."""
    p = OpenRouterProvider(api_key="or-test")
    assert p._build_extra_body({"provider_sort": "banana"}) == {}


def test_openrouter_build_extra_body_empty_or_none():
    p = OpenRouterProvider(api_key="or-test")
    assert p._build_extra_body(None) == {}
    assert p._build_extra_body({}) == {}


def test_openai_adapter_ignores_speed_options():
    """OpenAI direct has no sort knob; speed_options is dropped — hint is
    not a contract, so we silently proceed as default."""
    p = OpenAIProvider(api_key="sk-test")
    assert p._build_extra_body({"provider_sort": "latency"}) == {}


# ---------------------------------------------------------------------------
# Config loader: validation of the new speed_hints field
# ---------------------------------------------------------------------------


def _raw_config(speed_hints):
    return {
        "server": {"host": "127.0.0.1", "port": 1, "auth_secret": "s"},
        "providers": {"or": {"provider_type": "openrouter", "api_key": "k"}},
        "agents": {
            "a": {
                "provider": "or", "model": "m", "workspace": "/tmp/ws",
                "speed_hints": speed_hints,
            }
        },
    }


def test_config_loader_accepts_dict_and_null():
    cfg = _from_dict(_raw_config({
        "fast": {"provider_sort": "throughput"},
        "safe": None,
    }))
    agent = cfg.agents["a"]
    assert agent.speed_hints == {
        "fast": {"provider_sort": "throughput"},
        "safe": None,
    }


def test_config_loader_rejects_non_object_hints():
    with pytest.raises(ConfigError, match="speed_hints must be an object"):
        _from_dict(_raw_config([("fast", "latency")]))


def test_config_loader_rejects_scalar_hint_value():
    """A hint value must be an object (override) or null (disable). A bare
    string or number is a config mistake we surface rather than coerce."""
    with pytest.raises(ConfigError, match="speed_hints.fast"):
        _from_dict(_raw_config({"fast": "latency"}))


# ---------------------------------------------------------------------------
# End-to-end: the adapter actually puts `provider.sort` into the HTTP body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openrouter_sends_provider_sort_in_request(monkeypatch):
    """The whole point: when a turn carries `speed_options={provider_sort: latency}`,
    the OpenAI SDK client receives `extra_body={"provider": {"sort": "latency"}}`,
    which OpenRouter reads to bias provider selection."""
    from ark.types import AssistantTurnEnd, TurnUsageEvent, UserText

    captured: dict = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    p = OpenRouterProvider(api_key="or-test")
    p._client.chat = FakeChat()

    events = []
    async for evt in p.stream_turn(
        model="anthropic/claude-sonnet-4.5",
        system="",
        messages=[UserText(text="hi")],
        tools=[],
        speed_options={"provider_sort": "latency"},
    ):
        events.append(evt)

    assert captured["extra_body"] == {"provider": {"sort": "latency"}}
    # Everything else about the request is normal.
    assert captured["model"] == "anthropic/claude-sonnet-4.5"
    assert captured["stream"] is True


@pytest.mark.asyncio
async def test_openrouter_no_extra_body_when_no_hint(monkeypatch):
    """Default turns don't gain an extra_body — the hint is strictly opt-in
    per turn, so unrelated turns' requests stay byte-identical to today's."""
    from ark.types import UserText

    captured: dict = {}

    class FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    p = OpenRouterProvider(api_key="or-test")
    p._client.chat = FakeChat()

    async for _ in p.stream_turn(
        model="x/y",
        system="",
        messages=[UserText(text="hi")],
        tools=[],
    ):
        pass

    assert "extra_body" not in captured
