"""Real mid-turn cancel via the `stop` WS command (mine-capstone#485).

The turn task registers itself in runtime._active_turns; `stop` cancels it
(terminal `done {"stopped": true}` published by the task) and terminates
any in-flight run_command process group.
"""

import asyncio
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from ark import broker, db, runtime, tools
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import AssistantTurnEnd, TextDelta


def make_config(workspace):
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="m", workspace=workspace
            )
        },
    )


class _SlowThenFastProvider:
    """First stream hangs after one delta (until cancelled); later streams
    complete normally — lets one test prove both the cancel and that the
    session is immediately usable afterwards."""

    def __init__(self):
        self.calls = 0

    async def stream_turn(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            yield TextDelta(text="partial…")
            await asyncio.Event().wait()  # hang until cancelled
        else:
            yield TextDelta(text="ok")
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")


@pytest.fixture
def patched_provider(monkeypatch):
    provider = _SlowThenFastProvider()
    monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: provider)
    return provider


def _recv_until(ws, event_type, sid, tries=20):
    for _ in range(tries):
        frame = ws.receive_json()
        if frame.get("type") == event_type and frame.get("session_id") == sid:
            return frame
    raise AssertionError(f"never saw {event_type!r} for {sid!r}")


def test_stop_mid_stream_emits_done_stopped_and_session_stays_usable(
    ark_home, tmp_path, patched_provider
):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    app = create_app(make_config(ws_dir))
    client = TestClient(app)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    with client.websocket_connect("/events?token=x") as ws:
        ws.send_json({"type": "user_message", "session_id": sid, "text": "go"})
        delta = _recv_until(ws, "assistant_delta", sid)
        assert delta["text"] == "partial…"

        ws.send_json({"type": "stop", "session_id": sid})
        done = _recv_until(ws, "done", sid)
        assert done["stopped"] is True
        assert done["stop_reason"] == "stopped"
        assert runtime._active_turns == {}  # no orphan task handles

        # Session immediately usable: the next turn completes normally.
        ws.send_json({"type": "user_message", "session_id": sid, "text": "again"})
        message = _recv_until(ws, "assistant_message", sid)
        assert message["text"] == "ok"
        done2 = _recv_until(ws, "done", sid)
        assert done2["stop_reason"] == "end_turn"
        assert done2.get("stopped") is None


def test_stop_with_no_running_turn_is_a_silent_noop(ark_home, tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    app = create_app(make_config(ws_dir))
    client = TestClient(app)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    with client.websocket_connect("/events?token=x") as ws:
        ws.send_json({"type": "stop", "session_id": sid})
        # Prove no error frame arrived: an invalid command after it must be
        # the FIRST thing we receive.
        ws.send_json({"type": "bogus"})
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "unsupported command" in frame["message"]


def test_stop_without_session_id_is_an_error_frame(ark_home, tmp_path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    app = create_app(make_config(ws_dir))
    client = TestClient(app)

    with client.websocket_connect("/events?token=x") as ws:
        ws.send_json({"type": "stop"})
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "session_id" in frame["message"]


@pytest.mark.asyncio
async def test_stop_turn_cancels_registered_task(ark_home, tmp_path):
    started = asyncio.Event()

    async def fake_turn():
        runtime._active_turns["s1"] = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            runtime._active_turns.pop("s1", None)

    task = asyncio.create_task(fake_turn())
    await started.wait()
    assert runtime.stop_turn("s1") is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.stop_turn("s1") is False  # nothing left to cancel


@pytest.mark.asyncio
async def test_stop_session_commands_kills_the_process_group(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    ctx = tools.ToolContext(
        conn=None,
        config=cfg,
        agent=cfg.agents["scribe"],
        session_id="s-kill",
        cwd=tmp_path / "ws",
        loaded_skills=set(),
    )

    exec_task = asyncio.create_task(
        tools.execute("run_command", {"command": "sleep 30"}, ctx=ctx)
    )
    # Wait for the subprocess to register.
    deadline = time.monotonic() + 5
    while not tools._active_procs.get("s-kill"):
        if time.monotonic() > deadline:
            raise AssertionError("run_command never registered its process")
        await asyncio.sleep(0.02)

    assert tools.stop_session_commands("s-kill") == 1
    output, is_error = await asyncio.wait_for(exec_task, timeout=5)
    assert not is_error
    assert "exit code: -15" in output  # SIGTERM'd, well before the 30s sleep
    assert not tools._active_procs.get("s-kill")  # registry cleaned


@pytest.mark.asyncio
async def test_run_and_publish_registers_and_cleans_registry(
    ark_home, tmp_path, monkeypatch
):
    class _FastProvider:
        async def stream_turn(self, **kwargs):
            yield TextDelta(text="hi")
            yield AssistantTurnEnd(text="hi", stop_reason="end_turn")

    monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _FastProvider())
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    gq = broker.subscribe_all()
    try:
        await runtime.run_and_publish(
            conn=conn, config=cfg, agent=cfg.agents["scribe"],
            session_id=sid, user_text="hello",
        )
        assert runtime._active_turns == {}
    finally:
        broker.unsubscribe_all(gq)
