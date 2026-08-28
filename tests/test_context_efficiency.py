"""Context efficiency (mine-capstone#697 PR-1): named context blocks with
replace semantics, in-memory tool-result elision with hysteresis, cache
telemetry on TurnMetrics/TurnUsageEvent, and the /context endpoint."""

from fastapi.testclient import TestClient

from ark import runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.server import create_app
from ark.types import (
    SessionContext,
    ToolResult,
    TurnMetrics,
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
                model="claude-sonnet-4-6",
                workspace=workspace,
            )
        },
    )


H = {"Authorization": "Bearer x"}


def _rig(ark_home, tmp_path):
    cfg = make_config(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    app = create_app(cfg)
    return app, TestClient(app)


# ---------------------------------------------------------------------------
# named blocks
# ---------------------------------------------------------------------------


def test_named_context_roundtrips():
    role, body = message_to_row(SessionContext(text="focus v1", name="focus"))
    assert body == {"text": "focus v1", "name": "focus"}
    restored = message_from_row(role, body)
    assert restored.name == "focus"
    # Unnamed rows keep the legacy shape.
    _, legacy = message_to_row(SessionContext(text="plain"))
    assert legacy == {"text": "plain"}


def test_named_block_replaces_in_place_unnamed_stays_additive(tmp_path):
    agent = AgentConfig(name="scribe", provider="a", model="m", workspace=tmp_path)
    contexts = [
        SessionContext(text="FOCUS-V1", name="focus"),
        SessionContext(text="UNNAMED-A"),
        SessionContext(text="MODE-V1", name="mode"),
        SessionContext(text="FOCUS-V2", name="focus"),
        SessionContext(text="UNNAMED-B"),
    ]
    prompt = runtime.system_prompt(agent, contexts=contexts)
    # Last text at the FIRST position; one occurrence per name.
    assert "FOCUS-V1" not in prompt
    assert prompt.count("FOCUS-V2") == 1
    assert prompt.index("FOCUS-V2") < prompt.index("UNNAMED-A") < prompt.index("MODE-V1")
    assert "UNNAMED-A" in prompt and "UNNAMED-B" in prompt


def test_append_endpoint_reports_replaced(ark_home, tmp_path):
    app, client = _rig(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    first = client.post(
        f"/agents/scribe/sessions/{sid}/context", headers=H,
        json={"context": "focus v1", "name": "focus"},
    ).json()
    assert first == {"ok": True, "count": 1, "replaced": False}
    second = client.post(
        f"/agents/scribe/sessions/{sid}/context", headers=H,
        json={"context": "focus v2", "name": "focus"},
    ).json()
    assert second == {"ok": True, "count": 2, "replaced": True}
    unnamed = client.post(
        f"/agents/scribe/sessions/{sid}/context", headers=H,
        json={"context": "extra"},
    ).json()
    assert unnamed == {"ok": True, "count": 3, "replaced": False}


# ---------------------------------------------------------------------------
# tool-result elision
# ---------------------------------------------------------------------------


def _convo(result_bytes: int, results: int):
    messages = []
    for i in range(results):
        messages.append(UserText(text=f"turn {i}"))
        messages.append(ToolResult(
            call_id=f"c{i}", output="x" * result_bytes, is_error=False, name="big.tool",
        ))
    # The recent tail: two user turns, the last with its own big result —
    # inside the protected window, it must survive elision.
    messages.append(UserText(text="tail 0"))
    messages.append(UserText(text="tail 1"))
    messages.append(ToolResult(
        call_id="recent", output="y" * result_bytes, is_error=False, name="big.tool",
    ))
    return messages


def test_elision_hysteresis_below_threshold_untouched():
    messages = _convo(result_bytes=4096, results=3)
    out = runtime.elide_tool_results(messages, keep_turns=2, elide_over=2048,
                                     max_bytes=65536)
    assert out is messages  # 12 KB in view — nothing happens


def test_elision_one_pass_protects_recent_turns():
    messages = _convo(result_bytes=16384, results=6)  # 96 KB of results
    out = runtime.elide_tool_results(messages, keep_turns=2, elide_over=2048,
                                     max_bytes=65536)
    stubs = [m for m in out if isinstance(m, ToolResult) and "elided" in m.output]
    kept = [m for m in out if isinstance(m, ToolResult) and "elided" not in m.output]
    assert stubs and kept
    # The stub names the tool + call id and the original size.
    assert "big.tool" in stubs[0].output and "16384 bytes" in stubs[0].output
    # Results at/after the last 2 user turns stay whole.
    last_result = [m for m in out if isinstance(m, ToolResult)][-1]
    assert "elided" not in last_result.output
    # The ORIGINAL list is untouched (rows never mutate).
    assert all("elided" not in m.output for m in messages if isinstance(m, ToolResult))


# ---------------------------------------------------------------------------
# telemetry types
# ---------------------------------------------------------------------------


def test_turn_metrics_cache_fields_roundtrip():
    role, body = message_to_row(TurnMetrics(
        input_tokens=1000, output_tokens=10, model="m",
        cached_input_tokens=800, cache_write_tokens=50,
    ))
    restored = message_from_row(role, body)
    assert restored.cached_input_tokens == 800
    assert restored.cache_write_tokens == 50
    # Legacy rows (no cache fields) load as zeros.
    legacy = message_from_row("turn_metrics", {"input_tokens": 5, "output_tokens": 1})
    assert legacy.cached_input_tokens == 0


# ---------------------------------------------------------------------------
# GET /context
# ---------------------------------------------------------------------------


def test_context_endpoint_reports_blocks_and_usage(ark_home, tmp_path):
    app, client = _rig(ark_home, tmp_path)
    sid = client.post("/agents/scribe/sessions", headers=H).json()["id"]
    client.post(f"/agents/scribe/sessions/{sid}/context", headers=H,
                json={"context": "focus v1", "name": "focus"})
    client.post(f"/agents/scribe/sessions/{sid}/context", headers=H,
                json={"context": "unnamed"})
    client.post(f"/agents/scribe/sessions/{sid}/context", headers=H,
                json={"context": "focus v2 longer", "name": "focus"})
    runtime.append_message(app.state.conn, sid, TurnMetrics(
        input_tokens=5200, output_tokens=80, model="claude-sonnet-4-6",
        cached_input_tokens=4000,
    ))
    body = client.get(f"/agents/scribe/sessions/{sid}/context", headers=H).json()
    assert body["system_prompt_bytes"] > 0
    assert body["context_window"] == 200_000
    assert body["last_input_tokens"] == 5200
    assert body["last_cached_input_tokens"] == 4000
    assert body["compactions"] == 0
    blocks = body["blocks"]
    assert len(blocks) == 2  # focus (deduped) + unnamed
    focus = next(b for b in blocks if b["name"] == "focus")
    assert focus["bytes"] == len("focus v2 longer")
    assert focus["seq"] == 0  # first position kept
