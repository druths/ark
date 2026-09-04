"""Per-turn cumulative token budget — replaces the old max_iterations cap.

Budget precedence: explicit max_tokens arg > agent.max_turn_tokens > default.
Metric: cumulative input_tokens + output_tokens across TurnMetrics rows written
during THIS turn (compaction is excluded — no TurnMetrics emitted for it).
On breach: RunError(code='token_budget_exceeded') + RunEnd.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ark import config as _config, db, projects, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    RunEnd,
    RunError,
    RunErrorEvent,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
)


H = {"Authorization": "Bearer x"}


def make_cfg(tmp_path, **agent_overrides):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    agent = AgentConfig(
        name="scribe", provider="a", model="m", workspace=ws, **agent_overrides
    )
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={"scribe": agent},
    )


class _ScriptedProvider:
    """Each call in `scripts` is a list of provider events to emit that call.
    Later calls emit events from later scripts."""
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
        self.calls += 1
        script = self._scripts.pop(0) if self._scripts else []
        for evt in script:
            yield evt


def _factory(p):
    return lambda *_a, **_k: p


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _write_config(ark_home, data):
    (ark_home / "config.json").write_text(json.dumps(data))


def _min_cfg_data():
    return {
        "server": {"auth_secret": "shh"},
        "providers": {"a": {"provider_type": "anthropic", "api_key": "k"}},
        "agents": {"scribe": {"provider": "a", "model": "m"}},
    }


def test_agent_max_turn_tokens_optional_defaults_to_none(ark_home):
    _write_config(ark_home, _min_cfg_data())
    cfg = _config.load()
    assert cfg.agents["scribe"].max_turn_tokens is None


def test_agent_max_turn_tokens_parsed(ark_home):
    data = _min_cfg_data()
    data["agents"]["scribe"]["max_turn_tokens"] = 750_000
    _write_config(ark_home, data)
    cfg = _config.load()
    assert cfg.agents["scribe"].max_turn_tokens == 750_000


def test_agent_max_turn_tokens_non_positive_rejected(ark_home):
    data = _min_cfg_data()
    data["agents"]["scribe"]["max_turn_tokens"] = 0
    _write_config(ark_home, data)
    with pytest.raises(_config.ConfigError, match="positive integer"):
        _config.load()


def test_agent_max_turn_tokens_wrong_type_rejected(ark_home):
    data = _min_cfg_data()
    data["agents"]["scribe"]["max_turn_tokens"] = "big"
    _write_config(ark_home, data)
    with pytest.raises(_config.ConfigError, match="positive integer"):
        _config.load()


# ---------------------------------------------------------------------------
# Runtime: budget termination + precedence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_natural_turn_end_no_budget_breach(ark_home, tmp_path):
    """A turn that ends normally (no tool calls) never hits the budget check."""
    cfg = make_cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    provider = _ScriptedProvider([
        [TurnUsageEvent(input_tokens=100, output_tokens=50, model="m"),
         TextDelta(text="hi"),
         AssistantTurnEnd(text="hi", stop_reason="end")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hello",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    # Ran once, no runtime error, natural end:
    assert provider.calls == 1
    assert not any(isinstance(e, RunErrorEvent) for e in events)
    ends = [e for e in events if isinstance(e, RunEnd)]
    assert ends[-1].stop_reason == "end"


@pytest.mark.asyncio
async def test_budget_exceeded_after_iteration_terminates(ark_home, tmp_path):
    """Cumulative in+out exceeds budget after iteration 1 → runtime terminates
    with token_budget_exceeded before iteration 2."""
    cfg = make_cfg(tmp_path, max_turn_tokens=100)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    # Iteration 1 emits a tool call (so the loop would continue) AND enough
    # tokens to blow past the 100-token budget.
    provider = _ScriptedProvider([
        [TurnUsageEvent(input_tokens=80, output_tokens=40, model="m"),
         ToolCallEvent(id="t1", name="read_file", input={"path": "x"}),
         AssistantTurnEnd(text="", stop_reason="tool_use")],
        # Iteration 2 would send this if we got there — we shouldn't:
        [TurnUsageEvent(input_tokens=999, output_tokens=999, model="m"),
         AssistantTurnEnd(text="unreachable", stop_reason="end")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="please",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    # Only one provider call happened — the second script is untouched.
    assert provider.calls == 1
    errs = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(errs) == 1
    assert errs[0].code == "token_budget_exceeded"
    assert "120" in errs[0].message  # cumulative in+out
    assert "100" in errs[0].message  # budget

    ends = [e for e in events if isinstance(e, RunEnd)]
    assert ends[-1].stop_reason == "error:token_budget_exceeded"

    # RunError row is persisted.
    history = runtime.load_history(conn, sid)
    run_errors = [m for m in history if isinstance(m, RunError)]
    assert len(run_errors) == 1
    assert run_errors[0].code == "token_budget_exceeded"


@pytest.mark.asyncio
async def test_budget_check_is_post_iteration_first_always_runs(ark_home, tmp_path):
    """Even with a comically tight budget, iteration 1 always runs. The check
    only fires AFTER TurnMetrics lands, before iteration 2."""
    cfg = make_cfg(tmp_path, max_turn_tokens=1)  # tiny
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    provider = _ScriptedProvider([
        [TurnUsageEvent(input_tokens=100, output_tokens=100, model="m"),
         TextDelta(text="done"),
         AssistantTurnEnd(text="done", stop_reason="end")],  # no tool calls → natural end
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    # First iteration ran; no tool calls → natural end before the budget check.
    assert provider.calls == 1
    assert not any(isinstance(e, RunErrorEvent) for e in events)
    ends = [e for e in events if isinstance(e, RunEnd)]
    assert ends[-1].stop_reason == "end"


@pytest.mark.asyncio
async def test_explicit_max_tokens_arg_overrides_agent_setting(ark_home, tmp_path):
    """max_tokens explicitly passed to run_user_turn beats agent.max_turn_tokens."""
    cfg = make_cfg(tmp_path, max_turn_tokens=1_000_000)  # very generous per-agent
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    # Explicit override of 100 → iter 1 emits 200 tokens → breach.
    provider = _ScriptedProvider([
        [TurnUsageEvent(input_tokens=100, output_tokens=100, model="m"),
         ToolCallEvent(id="t1", name="read_file", input={"path": "x"}),
         AssistantTurnEnd(text="", stop_reason="tool_use")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
        max_tokens=100,
    ):
        events.append(evt)

    errs = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(errs) == 1
    assert errs[0].code == "token_budget_exceeded"


@pytest.mark.asyncio
async def test_default_budget_kicks_in_when_no_overrides(ark_home, tmp_path):
    """With neither an explicit arg nor an agent setting, the runtime falls
    back to DEFAULT_TURN_TOKEN_BUDGET."""
    from ark.runtime import DEFAULT_TURN_TOKEN_BUDGET
    assert DEFAULT_TURN_TOKEN_BUDGET >= 100_000  # sanity — expected to be generous

    cfg = make_cfg(tmp_path)  # no agent override
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    # Emit exactly DEFAULT+1 tokens in iter 1; assert breach.
    provider = _ScriptedProvider([
        [TurnUsageEvent(
            input_tokens=DEFAULT_TURN_TOKEN_BUDGET, output_tokens=1, model="m"
        ),
         ToolCallEvent(id="t1", name="x", input={}),
         AssistantTurnEnd(text="", stop_reason="tool_use")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    errs = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(errs) == 1
    assert errs[0].code == "token_budget_exceeded"


@pytest.mark.asyncio
async def test_multiple_iterations_within_budget_ok(ark_home, tmp_path):
    """Small iterations that stay under budget continue past what the old
    16-iteration cap allowed — this is the whole point of the change."""
    cfg = make_cfg(tmp_path, max_turn_tokens=10_000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    # 20 iterations, each ~200 tokens total. Old cap of 16 would have killed
    # this at iteration 16. New budget allows all 20.
    scripts = []
    for i in range(20):
        scripts.append([
            TurnUsageEvent(input_tokens=100, output_tokens=100, model="m"),
            ToolCallEvent(id=f"t{i}", name="x", input={}),
            AssistantTurnEnd(text="", stop_reason="tool_use"),
        ])
    # Final iteration ends naturally (no tool calls):
    scripts.append([
        TurnUsageEvent(input_tokens=50, output_tokens=50, model="m"),
        AssistantTurnEnd(text="fin", stop_reason="end"),
    ])
    provider = _ScriptedProvider(scripts)

    # Note: our runtime executes tool calls that are unknown as failing
    # ToolResults — the loop still continues.
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    # Reached the final iteration without a budget breach:
    assert provider.calls == 21
    assert not any(isinstance(e, RunErrorEvent) for e in events)
    ends = [e for e in events if isinstance(e, RunEnd)]
    assert ends[-1].stop_reason == "end"


# ---------------------------------------------------------------------------
# REST: PUT + GET for max_tokens
# ---------------------------------------------------------------------------


def _make_config_for_rest(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(name="scribe", provider="a", model="m", workspace=ws)
        },
    )


def test_put_cron_with_max_tokens(ark_home, tmp_path):
    client = TestClient(create_app(_make_config_for_rest(tmp_path)))
    conn = client.app.state.conn
    r = client.put(
        "/agents/scribe/crons/heavy",
        headers=H,
        json={"expr": "0 * * * *", "prompt": "big work", "max_tokens": 1_000_000},
    )
    assert r.status_code == 200, r.text
    row = conn.execute(
        "SELECT max_tokens FROM crons WHERE id = 'heavy'"
    ).fetchone()
    assert row["max_tokens"] == 1_000_000


def test_put_cron_max_tokens_omitted_preserves_existing(ark_home, tmp_path):
    """PUT without max_tokens must not clobber an existing per-cron budget —
    same shape as project_id."""
    client = TestClient(create_app(_make_config_for_rest(tmp_path)))
    conn = client.app.state.conn
    client.put(
        "/agents/scribe/crons/heavy", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": 750_000},
    )
    r = client.put(
        "/agents/scribe/crons/heavy", headers=H,
        json={"expr": "5 * * * *", "prompt": "different schedule"},
    )
    assert r.status_code == 200
    row = conn.execute(
        "SELECT expr, max_tokens FROM crons WHERE id = 'heavy'"
    ).fetchone()
    assert row["expr"] == "5 * * * *"
    assert row["max_tokens"] == 750_000  # preserved


def test_put_cron_max_tokens_null_reverts_to_default(ark_home, tmp_path):
    client = TestClient(create_app(_make_config_for_rest(tmp_path)))
    conn = client.app.state.conn
    client.put(
        "/agents/scribe/crons/heavy", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": 750_000},
    )
    r = client.put(
        "/agents/scribe/crons/heavy", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": None},
    )
    assert r.status_code == 200
    row = conn.execute(
        "SELECT max_tokens FROM crons WHERE id = 'heavy'"
    ).fetchone()
    assert row["max_tokens"] is None


def test_put_cron_max_tokens_invalid_400(ark_home, tmp_path):
    client = TestClient(create_app(_make_config_for_rest(tmp_path)))
    r = client.put(
        "/agents/scribe/crons/x", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": 0},
    )
    assert r.status_code == 400
    r = client.put(
        "/agents/scribe/crons/x", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": "big"},
    )
    assert r.status_code == 400


def test_get_crons_returns_max_tokens(ark_home, tmp_path):
    client = TestClient(create_app(_make_config_for_rest(tmp_path)))
    client.put(
        "/agents/scribe/crons/heavy", headers=H,
        json={"expr": "0 * * * *", "prompt": "x", "max_tokens": 500_000},
    )
    r = client.get("/agents/scribe/crons", headers=H)
    assert r.status_code == 200
    row = [c for c in r.json() if c["id"] == "heavy"][0]
    assert row["max_tokens"] == 500_000


# ---------------------------------------------------------------------------
# Scheduler precedence: cron.max_tokens beats agent.max_turn_tokens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_passes_cron_max_tokens_to_run_and_publish(
    ark_home, tmp_path, monkeypatch
):
    """The scheduler's _fire_cron should thread the cron row's max_tokens
    through to run_and_publish."""
    import ark.scheduler as sched_module

    cfg = make_cfg(tmp_path)
    conn = db.init_db()
    scheduler = sched_module.Scheduler(conn, cfg)

    seen: dict = {}

    async def _capture(**kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(runtime, "run_and_publish", _capture)
    await scheduler._fire_cron("scribe", "id1", "prompt", None, max_tokens=42_000)

    assert seen.get("max_tokens") == 42_000
