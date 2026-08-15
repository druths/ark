"""Per-session metadata (mine-capstone#528): stored server-side at session
create, surfaced to skills via ToolContext.metadata, and NEVER visible to
the model or to session read APIs.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from ark import db, runtime, tools
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import AssistantTurnEnd, TextDelta, ToolCallEvent

SECRET = "sekrit-callback-token"
META = {"mineai_gateway": {"url": "https://mineai.example", "secret": SECRET}}


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


def _make_client(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    app = create_app(make_config(ws))
    return TestClient(app)


AUTH = {"Authorization": "Bearer x"}


class TestCreateAndStorage:
    def test_metadata_stored_and_readable_server_side(self, ark_home, tmp_path):
        client = _make_client(tmp_path)
        conn = db.init_db()
        resp = client.post(
            "/agents/scribe/sessions", json={"metadata": META}, headers=AUTH
        )
        assert resp.status_code == 200
        sid = resp.json()["id"]
        assert runtime.session_metadata(conn, sid) == META

    def test_absent_metadata_reads_as_empty_dict(self, ark_home, tmp_path):
        client = _make_client(tmp_path)
        conn = db.init_db()
        sid = client.post("/agents/scribe/sessions", headers=AUTH).json()["id"]
        assert runtime.session_metadata(conn, sid) == {}

    def test_non_object_metadata_is_400(self, ark_home, tmp_path):
        client = _make_client(tmp_path)
        for bad in (["list"], "string", 42):
            resp = client.post(
                "/agents/scribe/sessions", json={"metadata": bad}, headers=AUTH
            )
            assert resp.status_code == 400, bad

    def test_metadata_not_exposed_by_read_apis(self, ark_home, tmp_path):
        """The pair is a capability, not content: session listings, detail
        and history must not leak it."""
        client = _make_client(tmp_path)
        sid = client.post(
            "/agents/scribe/sessions", json={"metadata": META}, headers=AUTH
        ).json()["id"]

        surfaces = [
            client.get("/agents/scribe/sessions", headers=AUTH),
            client.get(f"/sessions/{sid}", headers=AUTH),
            client.get(f"/agents/scribe/sessions/{sid}/history", headers=AUTH),
        ]
        for resp in surfaces:
            assert resp.status_code == 200
            assert SECRET not in resp.text


class TestTurnPlumbing:
    @pytest.mark.asyncio
    async def test_metadata_reaches_tool_context_but_never_the_model(
        self, ark_home, tmp_path, monkeypatch
    ):
        cfg = make_config(tmp_path / "ws")
        (tmp_path / "ws").mkdir()
        conn = db.init_db()
        sid = runtime.create_session(conn, "scribe", metadata=META)

        model_visible: list = []

        class _ToolCallingProvider:
            def __init__(self):
                self.calls = 0

            async def stream_turn(self, **kwargs):
                model_visible.append(
                    {"system": kwargs.get("system"), "messages": kwargs.get("messages")}
                )
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(id="c1", name="list_files", input={})
                    yield AssistantTurnEnd(text="", stop_reason="tool_use")
                else:
                    yield TextDelta(text="done")
                    yield AssistantTurnEnd(text="done", stop_reason="end_turn")

        provider = _ToolCallingProvider()
        monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: provider)

        seen_ctx: dict = {}
        real_execute = tools.execute

        async def capturing_execute(name, args, *, ctx):
            seen_ctx["metadata"] = ctx.metadata
            return await real_execute(name, args, ctx=ctx)

        monkeypatch.setattr(tools, "execute", capturing_execute)

        async for _ in runtime.run_user_turn(
            conn=conn, config=cfg, agent=cfg.agents["scribe"],
            session_id=sid, user_text="hello",
        ):
            pass

        assert seen_ctx["metadata"] == META  # skills see the capability...
        rendered = json.dumps(model_visible, default=str)
        assert SECRET not in rendered  # ...the model never does
        assert "mineai_gateway" not in rendered

    @pytest.mark.asyncio
    async def test_sessions_without_metadata_get_empty_dict_context(
        self, ark_home, tmp_path, monkeypatch
    ):
        cfg = make_config(tmp_path / "ws")
        (tmp_path / "ws").mkdir()
        conn = db.init_db()
        sid = runtime.create_session(conn, "scribe")

        class _Provider:
            def __init__(self):
                self.calls = 0

            async def stream_turn(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    yield ToolCallEvent(id="c1", name="list_files", input={})
                    yield AssistantTurnEnd(text="", stop_reason="tool_use")
                else:
                    yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

        monkeypatch.setattr(runtime, "make_provider", lambda *_a, **_k: _Provider())

        seen_ctx: dict = {}
        real_execute = tools.execute

        async def capturing_execute(name, args, *, ctx):
            seen_ctx["metadata"] = ctx.metadata
            return await real_execute(name, args, ctx=ctx)

        monkeypatch.setattr(tools, "execute", capturing_execute)

        async for _ in runtime.run_user_turn(
            conn=conn, config=cfg, agent=cfg.agents["scribe"],
            session_id=sid, user_text="hello",
        ):
            pass
        assert seen_ctx["metadata"] == {}
