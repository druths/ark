"""Scheduler firing semantics — heartbeats + crons, with attention to the
"added after startup" case that previously broke silently."""

import asyncio

import pytest

from ark import db
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.scheduler import Scheduler


def make_config(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    agent = AgentConfig(
        name="scribe", provider="a", model="m", workspace=ws
    )
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={"scribe": agent},
    )


@pytest.fixture
def sched(ark_home, tmp_path):
    """Scheduler with `_fire_cron` / `_fire_heartbeat` stubbed to record calls."""
    conn = db.init_db()
    s = Scheduler(conn, make_config(tmp_path))
    s.cron_fires: list[tuple[str, str]] = []
    s.heartbeat_fires: list[str] = []

    async def _fire_cron(agent_name, cron_id, prompt, project_id=None, max_tokens=None):
        s.cron_fires.append((agent_name, cron_id))

    async def _fire_heartbeat(agent_name):
        s.heartbeat_fires.append(agent_name)

    s._fire_cron = _fire_cron  # type: ignore[assignment]
    s._fire_heartbeat = _fire_heartbeat  # type: ignore[assignment]
    return s


async def _drain_tasks():
    """Yield once so any tasks spawned by _tick get to run."""
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# Anchor at a UTC minute boundary so `* * * * *` semantics are easy to reason
# about: next fire is always t0 + 60. 1_000_020 is 1970-01-12 13:47:00 UTC.
_MINUTE_ALIGNED_T0 = 1_000_020.0


@pytest.mark.asyncio
async def test_cron_added_at_runtime_fires(sched):
    """Regression: a cron added AFTER scheduler start used to never fire because
    `last_cron[key]` was never seeded — `_tick` reset the anchor to `now` on
    every pass via `.get(key, now)`. Confirmed broken in prod 2026-05-19."""

    sched.conn.execute(
        "INSERT INTO crons(agent_name, id, expr, prompt, enabled) "
        "VALUES (?, ?, ?, ?, 1)",
        ("scribe", "every-min", "* * * * *", "tick"),
    )

    t0 = _MINUTE_ALIGNED_T0
    await sched._tick(now=t0)
    await _drain_tasks()
    assert sched.cron_fires == []  # nothing fires — last_cron just seeded

    # 30 seconds in — still inside the same minute, shouldn't fire.
    await sched._tick(now=t0 + 30)
    await _drain_tasks()
    assert sched.cron_fires == []

    # 90 seconds in — we've crossed the next-minute boundary, fire once.
    await sched._tick(now=t0 + 90)
    await _drain_tasks()
    assert sched.cron_fires == [("scribe", "every-min")]


@pytest.mark.asyncio
async def test_cron_fires_repeatedly(sched):
    sched.conn.execute(
        "INSERT INTO crons(agent_name, id, expr, prompt, enabled) "
        "VALUES (?, ?, ?, ?, 1)",
        ("scribe", "every-min", "* * * * *", "tick"),
    )
    t0 = _MINUTE_ALIGNED_T0
    for offset in (0, 90, 180, 270):
        await sched._tick(now=t0 + offset)
        await _drain_tasks()
    # ticks at t0, t0+90, t0+180, t0+270 — three of those (90/180/270) crossed
    # a fresh minute boundary since the previous fire.
    assert len(sched.cron_fires) == 3


@pytest.mark.asyncio
async def test_disabled_cron_does_not_fire(sched):
    sched.conn.execute(
        "INSERT INTO crons(agent_name, id, expr, prompt, enabled) "
        "VALUES (?, ?, ?, ?, 0)",
        ("scribe", "off", "* * * * *", "tick"),
    )
    t0 = _MINUTE_ALIGNED_T0
    await sched._tick(now=t0)
    await sched._tick(now=t0 + 600)  # 10 minutes later
    await _drain_tasks()
    assert sched.cron_fires == []


@pytest.mark.asyncio
async def test_invalid_cron_expr_doesnt_kill_other_crons(sched):
    sched.conn.execute(
        "INSERT INTO crons(agent_name, id, expr, prompt, enabled) VALUES "
        "(?, ?, ?, ?, 1), (?, ?, ?, ?, 1)",
        (
            "scribe", "bad", "garbage cron", "x",
            "scribe", "good", "* * * * *", "y",
        ),
    )
    t0 = _MINUTE_ALIGNED_T0
    await sched._tick(now=t0)
    await sched._tick(now=t0 + 90)
    await _drain_tasks()
    assert sched.cron_fires == [("scribe", "good")]


@pytest.mark.asyncio
async def test_heartbeat_added_at_runtime_fires(sched):
    sched.conn.execute(
        "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, ?)",
        ("scribe", 60),
    )
    t0 = _MINUTE_ALIGNED_T0
    await sched._tick(now=t0)
    await _drain_tasks()
    assert sched.heartbeat_fires == []
    await sched._tick(now=t0 + 30)
    await _drain_tasks()
    assert sched.heartbeat_fires == []
    await sched._tick(now=t0 + 65)
    await _drain_tasks()
    assert sched.heartbeat_fires == ["scribe"]


@pytest.mark.asyncio
async def test_heartbeat_none_means_no_fire(sched):
    sched.conn.execute(
        "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, NULL)",
        ("scribe",),
    )
    t0 = _MINUTE_ALIGNED_T0
    for offset in (0, 60, 600, 3600):
        await sched._tick(now=t0 + offset)
    await _drain_tasks()
    assert sched.heartbeat_fires == []
