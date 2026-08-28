"""Client-invoked compaction: POST /agents/{name}/sessions/{sid}/compact.

Covers the two modes (server-generated + client-supplied), the pending-tool-call
guard, error surfaces, and the pending_tool_calls helper directly."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from ark import broker, db, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    CompactionSummary,
    TextDelta,
    ToolCall,
    ToolResult,
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
            "scribe": AgentConfig(name="scribe", provider="a", model="m", workspace=ws)
        },
    )


def _client(ark_home, tmp_path):
    return TestClient(create_app(make_config(tmp_path)))


# ---------------------------------------------------------------------------
# has_pending_tool_calls helper
# ---------------------------------------------------------------------------


def test_pending_when_toolcall_lacks_matching_result():
    history = [
        UserText(text="do it"),
        ToolCall(id="t1", name="x", input={}),
    ]
    assert runtime.has_pending_tool_calls(history) is True


def test_not_pending_when_toolcall_has_result():
    history = [
        UserText(text="do it"),
        ToolCall(id="t1", name="x", input={}),
        ToolResult(call_id="t1", output="ok", is_error=False, name="x"),
        AssistantText(text="done"),
    ]
    assert runtime.has_pending_tool_calls(history) is False


def test_not_pending_when_empty():
    assert runtime.has_pending_tool_calls([]) is False


# ---------------------------------------------------------------------------
# Client-supplied summary path (no LLM call)
# ---------------------------------------------------------------------------


def test_client_supplied_summary_persists_row_no_llm_call(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))
    runtime.append_message(conn, sid, AssistantText(text="hello"))

    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact",
        headers=H,
        json={"summary": "The user greeted me and I greeted back."},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["reason"] == "client-supplied"
    assert body["summary"] == "The user greeted me and I greeted back."

    history = runtime.load_history(conn, sid)
    summaries = [m for m in history if isinstance(m, CompactionSummary)]
    assert len(summaries) == 1
    assert summaries[0].text == "The user greeted me and I greeted back."
    assert summaries[0].reason == "client-supplied"


def test_client_supplied_summary_publishes_events(ark_home, tmp_path):
    """Even with no LLM call, WS clients should see the same lifecycle events
    they'd see for an auto compaction — so UI stays consistent."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))

    seen: list[dict] = []
    with patch.object(broker, "publish", side_effect=lambda _sid, ev: seen.append(ev)):
        r = client.post(
            f"/agents/scribe/sessions/{sid}/compact",
            headers=H,
            json={"summary": "x"},
        )
    assert r.status_code == 200
    types = [e.get("type") for e in seen]
    assert "compaction_started" in types
    assert "compaction_completed" in types
    for e in seen:
        assert e.get("session_id") == sid
        assert e.get("agent_name") == "scribe"


def test_empty_supplied_summary_rejected(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact",
        headers=H,
        json={"summary": "   "},
    )
    assert r.status_code == 400
    assert "non-empty" in r.text


def test_non_string_supplied_summary_rejected(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact",
        headers=H,
        json={"summary": 42},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Server-generated path
# ---------------------------------------------------------------------------


def test_server_generated_summary_calls_summarizer(ark_home, tmp_path, monkeypatch):
    """Empty body → server runs the summarizer and returns the generated text."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="teach me about pyramids"))
    runtime.append_message(conn, sid, AssistantText(text="pyramids are ..."))

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096, prompt_caching=False):
            # Verify the summarizer got the right shape.
            assert "producing a summary" in system.lower()
            assert tools == []
            yield TextDelta(text="Discussion of pyramids.")
            yield AssistantTurnEnd(text="Discussion of pyramids.", stop_reason="end")

    monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _StubProvider())

    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact", headers=H, json={}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["reason"] == "client-invoked"
    assert body["summary"] == "Discussion of pyramids."

    history = runtime.load_history(conn, sid)
    summaries = [m for m in history if isinstance(m, CompactionSummary)]
    assert len(summaries) == 1
    assert summaries[0].reason == "client-invoked"


def test_server_generated_summarizer_failure_returns_502(ark_home, tmp_path, monkeypatch):
    """When the summarizer raises, no CompactionSummary row is written and
    the client gets a 502 with the classified error."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))
    runtime.append_message(conn, sid, AssistantText(text="hello"))

    class _ExplodingProvider:
        async def stream_turn(self, **_kw):
            raise Exception("rate limit exceeded")
            yield  # pragma: no cover (generator hint)

    monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _ExplodingProvider())

    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact", headers=H, json={}
    )
    assert r.status_code == 502
    body = r.json()
    assert body["ok"] is False
    assert body["code"] == "rate_limit"

    history = runtime.load_history(conn, sid)
    assert not any(isinstance(m, CompactionSummary) for m in history)


def test_server_generated_empty_body_accepted(ark_home, tmp_path, monkeypatch):
    """A POST with no body at all is treated as `{}` (server-generated)."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))
    runtime.append_message(conn, sid, AssistantText(text="hello"))

    class _StubProvider:
        async def stream_turn(self, **_kw):
            yield TextDelta(text="s")
            yield AssistantTurnEnd(text="s", stop_reason="end")

    monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _StubProvider())

    r = client.post(f"/agents/scribe/sessions/{sid}/compact", headers=H)
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Guards: unknown / not-found / pending tool calls
# ---------------------------------------------------------------------------


def test_unknown_agent_404(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    r = client.post(
        "/agents/no-such-agent/sessions/any/compact", headers=H, json={"summary": "x"}
    )
    assert r.status_code == 404
    assert "unknown agent" in r.text


def test_unknown_session_404(ark_home, tmp_path):
    client = _client(ark_home, tmp_path)
    r = client.post(
        "/agents/scribe/sessions/no-such-sid/compact",
        headers=H, json={"summary": "x"},
    )
    assert r.status_code == 404
    assert "unknown session" in r.text


def test_pending_tool_call_blocks_compaction(ark_home, tmp_path):
    """Compacting across an unmatched ToolCall would leave the retry seeing
    a ToolResult referencing a call id it can't see. Reject with 409."""
    client = _client(ark_home, tmp_path)
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="do a thing"))
    runtime.append_message(conn, sid, ToolCall(id="t1", name="x", input={}))
    # No matching ToolResult → mid-tool-loop

    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact",
        headers=H, json={"summary": "x"},
    )
    assert r.status_code == 409
    assert "unmatched tool calls" in r.text


def test_compaction_ignores_enabled_flag(ark_home, tmp_path):
    """Manual endpoint runs even when compaction_enabled=false — the flag
    only gates AUTOMATIC triggers."""
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="m", workspace=ws,
                compaction_enabled=False,
            )
        },
    )
    client = TestClient(create_app(cfg))
    conn = client.app.state.conn
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    runtime.append_message(conn, sid, UserText(text="hi"))
    r = client.post(
        f"/agents/scribe/sessions/{sid}/compact",
        headers=H, json={"summary": "yes anyway"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
