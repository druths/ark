"""Per-session context: append-only client-supplied instructions."""

import io

import pytest
from fastapi.testclient import TestClient

from ark import db, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    AssistantText,
    SessionContext,
    ToolCall,
    ToolResult,
    UserText,
    message_from_row,
    message_to_row,
)


def make_config(workspace):
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"anthropic": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe",
                provider="anthropic",
                model="m",
                workspace=workspace,
            )
        },
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_session_context_roundtrips():
    role, body = message_to_row(SessionContext(text="be terse"))
    assert role == "session_context"
    assert body == {"text": "be terse"}
    restored = message_from_row(role, body)
    assert isinstance(restored, SessionContext)
    assert restored.text == "be terse"


# ---------------------------------------------------------------------------
# system_prompt composition
# ---------------------------------------------------------------------------


def test_system_prompt_no_contexts(tmp_path):
    agent = AgentConfig(name="scribe", provider="a", model="m", workspace=tmp_path)
    prompt = runtime.system_prompt(agent, contexts=[])
    # Layered sections present
    assert "scribe" in prompt
    assert "Environment" in prompt
    # No session context section yet
    assert "Session context" not in prompt


def test_system_prompt_with_contexts_appends_section(tmp_path):
    agent = AgentConfig(name="scribe", provider="a", model="m", workspace=tmp_path)
    contexts = [
        SessionContext(text="We're discussing Q4."),
        SessionContext(text="Alice is the project lead."),
    ]
    prompt = runtime.system_prompt(agent, contexts=contexts)
    assert "Session context" in prompt
    assert "We're discussing Q4." in prompt
    assert "Alice is the project lead." in prompt
    # The session-context section must come AFTER the Environment stanza
    assert prompt.index("Environment") < prompt.index("Session context")


def test_system_prompt_skips_empty_contexts(tmp_path):
    agent = AgentConfig(name="scribe", provider="a", model="m", workspace=tmp_path)
    prompt = runtime.system_prompt(
        agent, contexts=[SessionContext(text="   "), SessionContext(text="")]
    )
    # All whitespace/empty → no section emitted
    assert "Session context" not in prompt


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


def test_post_session_with_context_seeds_first_message(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    r = client.post(
        "/agents/scribe/sessions", headers=H, json={"context": "answer in haiku"}
    )
    assert r.status_code == 200
    sid = r.json()["id"]
    history = runtime.load_history(app.state.conn, sid)
    assert len(history) == 1
    assert isinstance(history[0], SessionContext)
    assert history[0].text == "answer in haiku"


def test_post_session_without_body_still_works(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    r = client.post("/agents/scribe/sessions", headers=H)  # no body
    assert r.status_code == 200
    assert runtime.load_history(app.state.conn, r.json()["id"]) == []


def test_post_context_appends_and_returns_count(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"context": "first"}
    ).json()["id"]
    r = client.post(
        f"/agents/scribe/sessions/{sid}/context", headers=H, json={"context": "second"}
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True, "count": 2, "replaced": False}
    history = runtime.load_history(app.state.conn, sid)
    contexts = [m for m in history if isinstance(m, SessionContext)]
    assert [c.text for c in contexts] == ["first", "second"]


def test_post_context_rejects_empty(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    for body in ({"context": ""}, {"context": "   "}, {}):
        r = client.post(
            f"/agents/scribe/sessions/{sid}/context", headers=H, json=body
        )
        assert r.status_code == 400


def test_post_context_unknown_session(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    r = client.post(
        "/agents/scribe/sessions/does-not-exist/context",
        headers=H,
        json={"context": "x"},
    )
    assert r.status_code == 404


def test_session_context_appears_in_history(ark_home, tmp_path):
    """Clients can inspect existing context via GET /history."""
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    client = TestClient(app)
    H = {"Authorization": "Bearer x"}
    sid = client.post(
        "/agents/scribe/sessions", headers=H, json={"context": "seed"}
    ).json()["id"]
    client.post(
        f"/agents/scribe/sessions/{sid}/context", headers=H, json={"context": "more"}
    )
    r = client.get(f"/agents/scribe/sessions/{sid}/history", headers=H)
    assert r.status_code == 200
    kinds = [m["kind"] for m in r.json()]
    assert kinds == ["SessionContext", "SessionContext"]


# ---------------------------------------------------------------------------
# run_user_turn filters context out of the LLM-facing turn list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_context_filtered_from_llm_messages(ark_home, tmp_path):
    """SessionContext rows must NOT be sent to the provider as turn messages —
    they belong in the system prompt only."""

    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, SessionContext(text="be terse"))

    captured = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096, prompt_caching=False):
            captured["system"] = system
            captured["messages"] = list(messages)
            from ark.types import AssistantTurnEnd

            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    def factory(*_args, **_kwargs):
        return _StubProvider()

    agent = cfg.agents["scribe"]
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn,
        config=cfg,
        agent=agent,
        session_id=sid,
        user_text="hi",
        provider_factory=factory,
    ):
        events.append(evt)

    # The provider must NOT have seen the SessionContext as a message:
    assert all(not isinstance(m, SessionContext) for m in captured["messages"])
    # It must have seen the user turn:
    assert any(isinstance(m, UserText) and m.text == "hi" for m in captured["messages"])
    # And the system prompt must include the context text:
    assert "be terse" in captured["system"]
