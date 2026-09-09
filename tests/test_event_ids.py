"""Live events for persisted rows carry `event_id` matching `messages.id`.

Feature: durable-cursor dedupe across the /events WS (live) and
GET /events (catch-up) — see docs/sessions.md § Event ids.

Contract:
- Persisted-row events → wire has `event_id` matching the row's messages.id.
- Ephemeral events (deltas, RunEnd, lifecycle-only compaction, workspace/
  project file changes) → wire has no `event_id`.
- For an event that has one, `live.event_id == catch_up.id` for the same row.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ark import broker, db, projects, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    CompactionCompletedEvent,
    RunError,
    RunErrorEvent,
    SharedFile,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
)


H = {"Authorization": "Bearer x"}


def make_config(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="claude-sonnet-4-6", workspace=ws
            )
        },
    )


def _client(ark_home, tmp_path):
    return TestClient(create_app(make_config(tmp_path)))


# ---------------------------------------------------------------------------
# append_message returns row id
# ---------------------------------------------------------------------------


def test_append_message_returns_row_id(ark_home, tmp_path):
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    id1 = runtime.append_message(conn, sid, UserText(text="one"))
    id2 = runtime.append_message(conn, sid, UserText(text="two"))
    assert isinstance(id1, int) and id1 > 0
    assert id2 == id1 + 1  # AUTOINCREMENT is monotonic


def test_returned_ids_match_history_ids(ark_home, tmp_path):
    """The value returned matches what /history and /events surface."""
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    returned = runtime.append_message(conn, sid, UserText(text="hello"))
    row = conn.execute(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (sid,),
    ).fetchone()
    assert row["id"] == returned


# ---------------------------------------------------------------------------
# event_to_wire: persisted-row events emit event_id when populated
# ---------------------------------------------------------------------------


def test_event_to_wire_assistant_message_carries_event_id():
    from ark.runtime import event_to_wire
    w = event_to_wire(AssistantTurnEnd(text="hi", stop_reason="end", row_id=42))
    assert w["type"] == "assistant_message"
    assert w["event_id"] == 42


def test_event_to_wire_assistant_message_omits_event_id_when_none():
    from ark.runtime import event_to_wire
    w = event_to_wire(AssistantTurnEnd(text="", stop_reason="end", row_id=None))
    assert w["type"] == "assistant_message"
    assert "event_id" not in w


def test_event_to_wire_tool_result_carries_event_id():
    from ark.runtime import event_to_wire
    w = event_to_wire(ToolResultEvent(call_id="t1", output="ok", is_error=False, row_id=99))
    assert w["event_id"] == 99
    assert w["id"] == "t1"  # tool-call correlation id preserved


def test_event_to_wire_tool_call_has_no_event_id():
    """Live tool_call frames go out before persistence — deliberately no
    event_id. Clients dedupe against the persisted-row events."""
    from ark.runtime import event_to_wire
    w = event_to_wire(ToolCallEvent(id="t1", name="x", input={}))
    assert "event_id" not in w


def test_event_to_wire_turn_usage_carries_event_id():
    from ark.runtime import event_to_wire
    w = event_to_wire(TurnUsageEvent(
        input_tokens=100, output_tokens=50, model="m", context_window=200000, row_id=17
    ))
    assert w["event_id"] == 17


def test_event_to_wire_run_error_carries_event_id():
    from ark.runtime import event_to_wire
    w = event_to_wire(RunErrorEvent(code="other", message="boom", row_id=7))
    assert w["event_id"] == 7


def test_event_to_wire_compaction_completed_carries_event_id():
    from ark.runtime import event_to_wire
    w = event_to_wire(CompactionCompletedEvent(summary="s", reason="auto", row_id=55))
    assert w["event_id"] == 55


def test_event_to_wire_ephemeral_events_have_no_event_id():
    """TextDelta, ThinkingDelta, RunEnd, and the lifecycle-only compaction
    frames all lack event_id — they don't correspond to a persisted row."""
    from ark.runtime import event_to_wire
    from ark.types import (
        CompactionFailedEvent,
        CompactionSkippedEvent,
        CompactionStartedEvent,
        RunEnd,
        ThinkingDelta,
    )
    for evt in (
        TextDelta(text="hi"),
        ThinkingDelta(text="hmm"),
        RunEnd(stop_reason="end"),
        CompactionStartedEvent(reason="x"),
        CompactionFailedEvent(code="other", message="boom", reason="x"),
        CompactionSkippedEvent(reason="disabled"),
    ):
        w = event_to_wire(evt)
        assert "event_id" not in w, f"unexpected event_id on {type(evt).__name__}"


# ---------------------------------------------------------------------------
# End-to-end: run_user_turn attaches ids that match persisted row ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_user_turn_ids_match_persisted_history(ark_home, tmp_path):
    """Every live event with event_id references a real row in messages;
    the value matches what /history + GET /events return for that row."""
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            yield TurnUsageEvent(input_tokens=100, output_tokens=50, model=model)
            yield TextDelta(text="answer")
            yield AssistantTurnEnd(text="answer", stop_reason="end_turn")

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hello",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        events.append(evt)

    # Find the ids the runtime attached to the persisted-row events.
    turn_end = next(e for e in events if isinstance(e, AssistantTurnEnd))
    usage = next(e for e in events if isinstance(e, TurnUsageEvent))
    assert turn_end.row_id is not None
    assert usage.row_id is not None

    # Both ids appear as real rows in messages.
    all_ids = [r["id"] for r in conn.execute(
        "SELECT id FROM messages WHERE session_id = ? ORDER BY id", (sid,)
    ).fetchall()]
    assert turn_end.row_id in all_ids
    assert usage.row_id in all_ids

    # And are distinct (different rows).
    assert turn_end.row_id != usage.row_id


@pytest.mark.asyncio
async def test_tool_result_gets_event_id_matching_row(ark_home, tmp_path):
    """The tool_result event's row_id matches the ToolResult row's messages.id."""
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    class _StubProvider:
        def __init__(self):
            self.n = 0

        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            self.n += 1
            if self.n == 1:
                yield TurnUsageEvent(input_tokens=50, output_tokens=10, model=model)
                yield ToolCallEvent(id="t1", name="unknown-tool", input={})
                yield AssistantTurnEnd(text="", stop_reason="tool_use")
            else:
                yield TurnUsageEvent(input_tokens=60, output_tokens=5, model=model)
                yield AssistantTurnEnd(text="done", stop_reason="end_turn")

    result_events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="try it",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        if isinstance(evt, ToolResultEvent):
            result_events.append(evt)

    assert len(result_events) == 1
    assert result_events[0].row_id is not None
    # And matches the actual ToolResult row.
    row = conn.execute(
        "SELECT id FROM messages WHERE session_id = ? AND role = 'tool_result'",
        (sid,),
    ).fetchone()
    assert row["id"] == result_events[0].row_id


# ---------------------------------------------------------------------------
# Broker publish sites (injected_message, file_available, session_project_changed)
# ---------------------------------------------------------------------------


def test_session_project_changed_publishes_event_id(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    root = tmp_path / "alpha"; root.mkdir()
    p = projects.create(
        client.app.state.conn,
        name="alpha", root=str(root), description="", project_context="",
    )
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        r = client.patch(
            f"/agents/scribe/sessions/{sid}/project",
            headers=H, json={"project_id": p.id},
        )
    assert r.status_code == 200
    e = next(e for e in seen if e.get("type") == "session_project_changed")
    assert isinstance(e["event_id"], int)


def test_client_supplied_compaction_publishes_event_id(ark_home, tmp_path):
    """Both endpoint publish sites (started + completed) fire; only
    completed carries event_id since only it persists a row."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        r = client.post(
            f"/agents/scribe/sessions/{sid}/compact",
            headers=H, json={"summary": "manual summary"},
        )
    assert r.status_code == 200
    started = next(e for e in seen if e.get("type") == "compaction_started")
    completed = next(e for e in seen if e.get("type") == "compaction_completed")
    assert "event_id" not in started      # lifecycle marker, not persisted
    assert isinstance(completed["event_id"], int)


# ---------------------------------------------------------------------------
# Outer catch: unhandled escapes now persist a RunError + carry event_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_and_publish_outer_catch_persists_and_carries_event_id(
    ark_home, tmp_path, monkeypatch
):
    """When something escapes run_user_turn (e.g. an event_to_wire bug), the
    outer catch persists a RunError so the error frame carries event_id."""
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        # Force an escape by making event_to_wire blow up.
        def _boom(_evt):
            raise RuntimeError("wire conversion failed")

        monkeypatch.setattr(runtime, "event_to_wire", _boom)

        class _StubProvider:
            async def stream_turn(self, **_kw):
                yield TextDelta(text="hi")
                yield AssistantTurnEnd(text="hi", stop_reason="end_turn")

        monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _StubProvider())
        await runtime.run_and_publish(
            conn=conn, config=cfg, agent=cfg.agents["scribe"],
            session_id=sid, user_text="hi",
        )

    err = next(e for e in seen if e.get("type") == "error")
    assert err["code"] == "other"
    assert isinstance(err["event_id"], int)

    # And the RunError row is really in history.
    from ark.types import RunError as _RE
    history = runtime.load_history(conn, sid)
    assert any(isinstance(m, _RE) for m in history)


# ---------------------------------------------------------------------------
# Catch-up alignment: /events?since_id returns rows with the same ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_event_id_matches_catch_up_id(ark_home, tmp_path):
    """The whole point of the feature: live wire event's event_id is the
    same integer /events (catch-up) returns for that row."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            yield TurnUsageEvent(input_tokens=10, output_tokens=5, model=model)
            yield TextDelta(text="hi")
            yield AssistantTurnEnd(text="hi", stop_reason="end_turn")

    with patch.object(runtime, "make_provider", lambda *_a, **_k: _StubProvider()):
        sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
        # Capture broker events.
        seen: list[dict] = []
        with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
            await runtime.run_and_publish(
                conn=conn,
                config=client.app.state.config,
                agent=client.app.state.config.agents["scribe"],
                session_id=sid, user_text="hi",
            )

        # Find the assistant_message event on the wire.
        am = next(e for e in seen if e.get("type") == "assistant_message")
        live_id = am["event_id"]

        # Now fetch via /events — the AssistantText row should have this id.
        r = client.get(f"/events?since_id={live_id - 1}&limit=1000", headers=H)
        events = r.json()["events"]
        match = next(e for e in events if e["id"] == live_id and e["kind"] == "AssistantText")
        assert match["data"]["text"] == "hi"
