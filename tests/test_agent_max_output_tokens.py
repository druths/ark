"""Per-agent max_output_tokens config (per-response output cap on the SDK
`max_tokens` kwarg passed to stream_turn). Distinct from `max_turn_tokens`
(whole-turn budget) and cron `max_tokens` (per-cron turn budget)."""

import json

import pytest

from ark import config


def write(d, data):
    (d / "config.json").write_text(json.dumps(data))


def minimal(agent_extra=None):
    return {
        "server": {"auth_secret": "shh"},
        "providers": {"anthropic": {"provider_type": "anthropic", "api_key": "k"}},
        "agents": {
            "scribe": {
                "provider": "anthropic",
                "model": "claude-opus-4-7",
                **(agent_extra or {}),
            }
        },
    }


def test_max_output_tokens_defaults_to_none(ark_home):
    write(ark_home, minimal())
    assert config.load().agents["scribe"].max_output_tokens is None


def test_max_output_tokens_parsed(ark_home):
    write(ark_home, minimal({"max_output_tokens": 8192}))
    assert config.load().agents["scribe"].max_output_tokens == 8192


@pytest.mark.parametrize("bad", [0, -1, "8192", 1.5])
def test_max_output_tokens_rejects_non_positive_ints(ark_home, bad):
    write(ark_home, minimal({"max_output_tokens": bad}))
    with pytest.raises(config.ConfigError, match="max_output_tokens"):
        config.load()


# ---------------------------------------------------------------------------
# End-to-end: the value actually reaches provider.stream_turn(max_tokens=...)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_output_tokens_flows_to_provider(ark_home, tmp_path):
    from ark import db, runtime
    from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
    from ark.types import AssistantTurnEnd, TextDelta

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="m", workspace=ws,
                max_output_tokens=16000,
            )
        },
    )
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    seen: dict = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            seen["max_tokens"] = max_tokens
            yield TextDelta(text="hi")
            yield AssistantTurnEnd(text="hi", stop_reason="end")

    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hello",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        pass
    assert seen["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_max_output_tokens_defaults_to_4096_when_none(ark_home, tmp_path):
    """Absent from config → provider gets the 4096 fallback (matching every
    provider adapter's def-time default). Backwards-compatible with pre-PR
    behavior."""
    from ark import db, runtime
    from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
    from ark.types import AssistantTurnEnd, TextDelta

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(name="scribe", provider="a", model="m", workspace=ws)
        },
    )
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    seen: dict = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            seen["max_tokens"] = max_tokens
            yield TextDelta(text="hi")
            yield AssistantTurnEnd(text="hi", stop_reason="end")

    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hello",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        pass
    assert seen["max_tokens"] == 4096
