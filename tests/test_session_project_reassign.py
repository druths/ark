"""Session ↔ project reassignment: PATCH /agents/{name}/sessions/{sid}/project,
the ProjectAssignmentChanged marker, and the runtime substitution that turns
the marker into a synthetic user notification for the LLM."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ark import broker, db, projects, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    ProjectAssignmentChanged,
    ToolCall,
    UserText,
    message_from_row,
    message_to_row,
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
            "scribe": AgentConfig(name="scribe", provider="a", model="m", workspace=ws)
        },
    )


def _client(ark_home, tmp_path):
    return TestClient(create_app(make_config(tmp_path)))


def _make_project(client, name: str, tmp_path: Path):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    r = client.post("/projects", headers=H, json={"name": name, "root": str(root)})
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_project_assignment_changed_round_trip():
    m = ProjectAssignmentChanged(
        from_project_id="a", to_project_id="b",
        from_project_name="alpha", to_project_name="beta",
        from_root="/tmp/a", to_root="/tmp/b",
        changed_at=1234,
    )
    role, content = message_to_row(m)
    assert role == "project_assignment_changed"
    restored = message_from_row(role, content)
    assert isinstance(restored, ProjectAssignmentChanged)
    assert restored.from_project_id == "a"
    assert restored.to_project_id == "b"
    assert restored.changed_at == 1234


# ---------------------------------------------------------------------------
# Endpoint: happy paths
# ---------------------------------------------------------------------------


def test_assign_previously_unassigned(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    pid = _make_project(client, "alpha", tmp_path)

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": pid},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["changed"] is True
    assert body["from"] is None
    assert body["to"]["id"] == pid

    # DB reflects the change:
    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["project_id"] == pid

    # History has the marker:
    history = runtime.load_history(conn, sid)
    markers = [m for m in history if isinstance(m, ProjectAssignmentChanged)]
    assert len(markers) == 1
    assert markers[0].from_project_id is None
    assert markers[0].to_project_id == pid
    assert markers[0].to_project_name == "alpha"


def test_reassign_between_projects(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    p1 = _make_project(client, "alpha", tmp_path)
    p2 = _make_project(client, "beta", tmp_path)
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"project_id": p1}
    ).json()["id"]

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": p2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["from"]["id"] == p1
    assert body["to"]["id"] == p2

    history = runtime.load_history(conn, sid)
    markers = [m for m in history if isinstance(m, ProjectAssignmentChanged)]
    assert len(markers) == 1
    assert markers[0].from_project_id == p1
    assert markers[0].to_project_id == p2
    assert markers[0].from_project_name == "alpha"
    assert markers[0].to_project_name == "beta"


def test_detach_from_project(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    p1 = _make_project(client, "alpha", tmp_path)
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"project_id": p1}
    ).json()["id"]

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["from"]["id"] == p1
    assert body["to"] is None

    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    assert row["project_id"] is None


def test_reassign_idempotent_no_op(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    p1 = _make_project(client, "alpha", tmp_path)
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"project_id": p1}
    ).json()["id"]

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": p1},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["changed"] is False

    # No marker was written:
    history = runtime.load_history(conn, sid)
    assert not any(isinstance(m, ProjectAssignmentChanged) for m in history)


def test_detach_when_already_unassigned_is_no_op(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": None},
    )
    assert r.status_code == 200
    assert r.json()["changed"] is False
    history = runtime.load_history(conn, sid)
    assert not any(isinstance(m, ProjectAssignmentChanged) for m in history)


def test_endpoint_publishes_broker_event(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    p1 = _make_project(client, "alpha", tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        r = client.patch(
            f"/agents/scribe/sessions/{sid}/project",
            headers=H,
            json={"project_id": p1},
        )
    assert r.status_code == 200
    change_events = [e for e in seen if e.get("type") == "session_project_changed"]
    assert len(change_events) == 1
    e = change_events[0]
    assert e["session_id"] == sid
    assert e["agent_name"] == "scribe"
    assert e["from_project_id"] is None
    assert e["to_project_id"] == p1
    assert e["to_project_name"] == "alpha"


def test_no_broker_event_on_noop(ark_home, tmp_path):
    """A no-op PATCH must not publish an event — clients would misread it as
    a real transition."""
    client = _client(ark_home, tmp_path)
    p1 = _make_project(client, "alpha", tmp_path)
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"project_id": p1}
    ).json()["id"]

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        r = client.patch(
            f"/agents/scribe/sessions/{sid}/project",
            headers=H,
            json={"project_id": p1},
        )
    assert r.status_code == 200
    change_events = [e for e in seen if e.get("type") == "session_project_changed"]
    assert change_events == []


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_unknown_agent_404(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    r = client.patch(
        "/agents/no-such-agent/sessions/x/project",
        headers=H, json={"project_id": None},
    )
    assert r.status_code == 404
    assert "unknown agent" in r.text


def test_unknown_session_404(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    r = client.patch(
        "/agents/scribe/sessions/no-such-sid/project",
        headers=H, json={"project_id": None},
    )
    assert r.status_code == 404
    assert "unknown session" in r.text


def test_unknown_project_404(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": "no-such-project"},
    )
    assert r.status_code == 404
    assert "unknown project" in r.text


def test_soft_deleted_project_rejected(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    p1 = _make_project(client, "alpha", tmp_path)
    client.delete(f"/projects/{p1}", headers=H)  # soft-delete
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": p1},
    )
    assert r.status_code == 404


def test_pending_tool_call_blocks_reassign(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    p1 = _make_project(client, "alpha", tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    # Session in mid-tool-loop: unmatched ToolCall.
    runtime.append_message(conn, sid, UserText(text="do it"))
    runtime.append_message(conn, sid, ToolCall(id="t1", name="x", input={}))

    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": p1},
    )
    assert r.status_code == 409
    assert "unmatched tool calls" in r.text


def test_missing_project_id_field_400(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={},
    )
    assert r.status_code == 400


def test_non_string_project_id_400(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.patch(
        f"/agents/scribe/sessions/{sid}/project",
        headers=H,
        json={"project_id": 42},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# LLM view: the marker becomes a synthetic UserText notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_sees_synthetic_notification_and_new_project_in_sys_prompt(
    ark_home, tmp_path
):
    """After reassignment: the model sees the transition as a UserText at
    that point in the message list AND the system prompt reflects the new
    project's stanza."""
    from ark.types import TextDelta

    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    # Set up a project + prior turn:
    root = tmp_path / "beta"; root.mkdir()
    pid = projects.create(
        conn, name="beta", root=str(root), description="", project_context=""
    ).id

    runtime.append_message(conn, sid, UserText(text="hello"))
    runtime.append_message(conn, sid, AssistantText(text="hi"))

    # Simulate reassignment.
    runtime.set_session_project(conn, sid, pid)

    class _StubProvider:
        def __init__(self):
            self.calls = []
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096, prompt_caching=False):
            self.calls.append({"system": system, "messages": list(messages)})
            yield TextDelta(text="ack")
            yield AssistantTurnEnd(text="ack", stop_reason="end")

    provider = _StubProvider()
    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="continue",
        provider_factory=lambda *_a, **_k: provider,
    ):
        pass

    seen = provider.calls[0]
    # System prompt has the new project stanza:
    assert "beta" in seen["system"]
    assert str(root) in seen["system"]

    # Message list contains the synthetic transition notification substituted
    # for the marker. The raw marker itself must NOT appear.
    for m in seen["messages"]:
        assert not isinstance(m, ProjectAssignmentChanged)
    texts = [
        m.text for m in seen["messages"]
        if isinstance(m, UserText)
    ]
    joined = "\n".join(texts)
    assert "system notification" in joined
    assert "project assignment changed" in joined
    assert "beta" in joined
    # And user turns and assistant turns are still in order:
    assert texts[0] == "hello"        # original user turn
    # The new user turn we sent is the last UserText:
    assert texts[-1] == "continue"
