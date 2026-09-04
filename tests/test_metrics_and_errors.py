"""Per-turn usage metrics + classified error tracking."""

import asyncio

import pytest

from ark import db, models, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.runtime import classify_provider_error
from ark.types import (
    AssistantTurnEnd,
    RunError,
    RunErrorEvent,
    SessionContext,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
    message_from_row,
    message_to_row,
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_turn_metrics_roundtrip():
    role, body = message_to_row(TurnMetrics(input_tokens=1234, output_tokens=89, model="m"))
    assert role == "turn_metrics"
    assert body == {"input_tokens": 1234, "output_tokens": 89, "model": "m"}
    assert message_from_row(role, body) == TurnMetrics(1234, 89, "m")


def test_run_error_roundtrip():
    role, body = message_to_row(RunError(code="context_too_long", message="boom"))
    assert role == "run_error"
    assert body == {"code": "context_too_long", "message": "boom"}
    assert message_from_row(role, body) == RunError("context_too_long", "boom")


# ---------------------------------------------------------------------------
# Model context-window lookup
# ---------------------------------------------------------------------------


def test_context_window_lookup_known_model():
    assert models.context_window_for("claude-sonnet-4-6") == 200_000
    assert models.context_window_for("gemini-2.5-pro") == 1_048_576


def test_context_window_lookup_unknown():
    assert models.context_window_for("imaginary-model-9000") is None


def test_context_window_override_wins():
    assert models.context_window_for("claude-sonnet-4-6", override=500_000) == 500_000


def test_context_window_strips_openrouter_prefix():
    assert (
        models.context_window_for("anthropic/claude-sonnet-4-6") == 200_000
    )


# ---------------------------------------------------------------------------
# classify_provider_error
# ---------------------------------------------------------------------------


def test_classify_context_too_long_variants():
    for msg in (
        "context_length_exceeded: 200001 > 200000",
        "prompt is too long: 250000 tokens",
        "Input is too long for the model's context window",
        "Request exceeds the maximum context length of 200000 tokens",
    ):
        code, _ = classify_provider_error(ValueError(msg))
        assert code == "context_too_long", msg


def test_classify_rate_limit():
    class RateLimitError(Exception):
        pass

    assert classify_provider_error(RateLimitError("slow down"))[0] == "rate_limit"
    assert classify_provider_error(Exception("429 too many requests"))[0] == "rate_limit"


def test_classify_auth():
    class AuthenticationError(Exception):
        pass

    assert classify_provider_error(AuthenticationError("nope"))[0] == "auth"
    assert classify_provider_error(Exception("401 Unauthorized"))[0] == "auth"
    assert classify_provider_error(Exception("invalid api key"))[0] == "auth"


def test_classify_other_fallback():
    assert classify_provider_error(Exception("server hiccup"))[0] == "other"


# ---------------------------------------------------------------------------
# Message enrichment: class name + status_code + request_id
# ---------------------------------------------------------------------------


def test_classify_prepends_exception_class_name():
    """Even for the "other" bucket, the class name should carry through so
    a bare 'Internal Server Error' becomes 'InternalServerError: Internal
    Server Error' — enough to tell it's a provider 5xx vs. a local bug."""

    class InternalServerError(Exception):
        pass

    code, msg = classify_provider_error(InternalServerError("Internal Server Error"))
    assert code == "other"
    assert msg.startswith("InternalServerError:")
    assert "Internal Server Error" in msg


def test_classify_includes_status_code_when_present():
    """SDK exceptions like anthropic.APIStatusError expose .status_code —
    include it so operators can distinguish a 500 from a 502 without a stack."""

    class APIStatusError(Exception):
        def __init__(self, message, status_code):
            super().__init__(message)
            self.status_code = status_code

    _, msg = classify_provider_error(APIStatusError("boom", 500))
    assert "[status=500]" in msg
    assert "boom" in msg


def test_classify_includes_request_id_when_present():
    """Direct .request_id attribute (OpenAI/Anthropic SDKs) is surfaced."""

    class APIError(Exception):
        pass

    exc = APIError("bad happened")
    exc.request_id = "req_abc123"
    _, msg = classify_provider_error(exc)
    assert msg.endswith("(request_id=req_abc123)")


def test_classify_extracts_request_id_from_response_headers():
    """When request_id isn't a direct attribute, fall back to response
    headers — where Anthropic and OpenAI actually stash it on their
    APIStatusError.response objects."""

    class _Response:
        headers = {"x-request-id": "req_headerish"}

    class APIStatusError(Exception):
        def __init__(self, msg):
            super().__init__(msg)
            self.status_code = 502
            self.response = _Response()

    _, msg = classify_provider_error(APIStatusError("upstream 502"))
    assert "[status=502]" in msg
    assert "(request_id=req_headerish)" in msg


def test_classify_no_request_id_no_status_still_carries_class():
    """Baseline: even with a plain Exception the message includes the class."""
    _, msg = classify_provider_error(Exception("boring failure"))
    assert msg.startswith("Exception: boring failure")


def test_classify_empty_message_still_useful():
    """Some providers raise with no message body — the class name alone
    is still better than the pre-enrichment blank."""

    class TimeoutError(Exception):  # noqa: A001
        pass

    _, msg = classify_provider_error(TimeoutError())
    assert msg == "TimeoutError"


# ---------------------------------------------------------------------------
# Runtime: usage is persisted, errors are caught + persisted, both filtered
# from the LLM message list on subsequent iterations.
# ---------------------------------------------------------------------------


def make_config(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    agent = AgentConfig(
        name="scribe", provider="a", model="claude-sonnet-4-6", workspace=ws
    )
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={"scribe": agent},
    )


@pytest.mark.asyncio
async def test_turn_usage_is_persisted_and_emitted(ark_home, tmp_path):
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            yield TurnUsageEvent(input_tokens=1234, output_tokens=42, model=model)
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn,
        config=cfg,
        agent=cfg.agents["scribe"],
        session_id=sid,
        user_text="hi",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        events.append(evt)

    # Emitted to the client with context_window resolved from the model table:
    usage_events = [e for e in events if isinstance(e, TurnUsageEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].input_tokens == 1234
    assert usage_events[0].context_window == 200_000  # claude-sonnet-4-6

    # Persisted to history as TurnMetrics:
    history = runtime.load_history(conn, sid)
    metrics = [m for m in history if isinstance(m, TurnMetrics)]
    assert len(metrics) == 1
    assert metrics[0].input_tokens == 1234
    assert metrics[0].output_tokens == 42


@pytest.mark.asyncio
async def test_turn_metrics_filtered_from_llm_list(ark_home, tmp_path):
    """TurnMetrics rows must not be sent to the provider as conversation turns."""
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=999, output_tokens=10))

    captured = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            captured["messages"] = list(messages)
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    async for _ in runtime.run_user_turn(
        conn=conn,
        config=cfg,
        agent=cfg.agents["scribe"],
        session_id=sid,
        user_text="hi",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        pass

    assert all(not isinstance(m, TurnMetrics) for m in captured["messages"])
    # And the user message is in the list:
    assert any(isinstance(m, UserText) and m.text == "hi" for m in captured["messages"])


@pytest.mark.asyncio
async def test_provider_exception_caught_classified_and_persisted(ark_home, tmp_path):
    cfg = make_config(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    class _ExplodingProvider:
        async def stream_turn(self, **_kwargs):
            raise RuntimeError("prompt is too long: 250000 tokens")
            yield  # unreachable but makes this an async generator

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn,
        config=cfg,
        agent=cfg.agents["scribe"],
        session_id=sid,
        user_text="hi",
        provider_factory=lambda *_a, **_k: _ExplodingProvider(),
    ):
        events.append(evt)

    err_events = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(err_events) == 1
    assert err_events[0].code == "context_too_long"

    # Persisted in history so subsequent /history reads see it:
    history = runtime.load_history(conn, sid)
    errors = [m for m in history if isinstance(m, RunError)]
    assert len(errors) == 1
    assert errors[0].code == "context_too_long"
