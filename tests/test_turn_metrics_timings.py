"""Per-call wall-clock timings on TurnMetrics / TurnUsageEvent.

Each provider adapter is responsible for stamping TTFT (request → first
streamed output event) and total duration (request → usage chunk) on the
TurnUsageEvent it yields. The runtime copies those fields onto the
persisted TurnMetrics row and the forwarded wire event.

Uses a controllable fake monotonic clock so timings are deterministic
rather than real wall-clock.
"""

from unittest.mock import patch

import pytest

from ark import provider as _provider_mod
from ark.provider import (
    AnthropicProvider,
    OpenAIProvider,
    OpenRouterProvider,
    _with_timings,
)
from ark.types import (
    AssistantTurnEnd,
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
    message_from_row,
    message_to_row,
)


# ---------------------------------------------------------------------------
# Serialization: TurnMetrics round-trips the new fields; omits defaults
# ---------------------------------------------------------------------------


def test_turn_metrics_round_trip_without_timings():
    """Rows written by code that doesn't know about timings (or by the
    runtime when the adapter didn't stamp them) stay byte-identical to the
    legacy shape on disk — no stray `ttft_ms: null` keys."""
    m = TurnMetrics(input_tokens=10, output_tokens=20, model="x")
    role, body = message_to_row(m)
    assert role == "turn_metrics"
    assert "ttft_ms" not in body
    assert "duration_ms" not in body
    restored = message_from_row(role, body)
    assert restored == TurnMetrics(input_tokens=10, output_tokens=20, model="x")
    assert restored.ttft_ms is None
    assert restored.duration_ms is None


def test_turn_metrics_round_trip_with_timings():
    m = TurnMetrics(
        input_tokens=10, output_tokens=20, model="x",
        ttft_ms=42.5, duration_ms=180.0,
    )
    role, body = message_to_row(m)
    assert body["ttft_ms"] == 42.5
    assert body["duration_ms"] == 180.0
    restored = message_from_row(role, body)
    assert restored == m


def test_turn_metrics_loads_legacy_row_unchanged():
    """Pre-timing-era row in the DB: no ttft/duration keys. Must load as a
    TurnMetrics with None timings, not crash on a missing key."""
    legacy_body = {"input_tokens": 10, "output_tokens": 20, "model": "x"}
    restored = message_from_row("turn_metrics", legacy_body)
    assert restored == TurnMetrics(input_tokens=10, output_tokens=20, model="x")
    assert restored.ttft_ms is None
    assert restored.duration_ms is None


# ---------------------------------------------------------------------------
# _with_timings: the single stamping helper used by every adapter
# ---------------------------------------------------------------------------


def test_with_timings_populates_both_fields():
    """Normal case: TTFT and duration are both captured. Monotonic deltas
    in seconds become ms on the event."""
    evt = TurnUsageEvent(input_tokens=1, output_tokens=2)
    with patch.object(_provider_mod.time, "monotonic", return_value=5.0):
        out = _with_timings(evt, t0=4.5, t_first=4.6)
    assert out.ttft_ms == pytest.approx(100.0)   # 4.6 - 4.5
    assert out.duration_ms == pytest.approx(500.0)  # 5.0 - 4.5
    # Other fields untouched.
    assert out.input_tokens == 1
    assert out.output_tokens == 2


def test_with_timings_no_output_leaves_ttft_none():
    """Rare but real: a stream that ends with no TextDelta/Thinking/ToolCall
    (e.g. an immediate stop). We can still report duration but TTFT is
    None, not 0 — 0 would look like a <1ms response and mislead dashboards."""
    evt = TurnUsageEvent(input_tokens=1, output_tokens=0)
    with patch.object(_provider_mod.time, "monotonic", return_value=5.0):
        out = _with_timings(evt, t0=4.5, t_first=None)
    assert out.ttft_ms is None
    assert out.duration_ms == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Adapter integration: each stream_turn stamps timings via a fake stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_adapter_stamps_ttft_and_duration():
    """Drive the OpenAI adapter's stream_turn end-to-end with a fake chat
    completions stream + fake monotonic clock; the final TurnUsageEvent
    must carry the computed TTFT/duration."""
    import json as _json
    from types import SimpleNamespace

    # Fake OpenAI chat-completions stream: one content chunk, one usage chunk.
    class FakeStream:
        def __init__(self):
            self._items = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="hi", tool_calls=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content=None, tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=7, completion_tokens=11),
                ),
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    # Each call to monotonic() advances: t0, first-token, then end.
    clock = iter([100.0, 100.25, 100.9])

    class FakeCompletions:
        async def create(self, **_kw):
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    p = OpenAIProvider(api_key="sk")
    p._client.chat = FakeChat()

    with patch.object(_provider_mod.time, "monotonic", side_effect=lambda: next(clock)):
        events = [
            e async for e in p.stream_turn(
                model="gpt-4o",
                system="",
                messages=[UserText(text="hi")],
                tools=[],
            )
        ]

    usage = [e for e in events if isinstance(e, TurnUsageEvent)]
    assert len(usage) == 1
    assert usage[0].ttft_ms == pytest.approx(250.0)   # 100.25 - 100.0
    assert usage[0].duration_ms == pytest.approx(900.0)  # 100.9 - 100.0
    assert usage[0].input_tokens == 7
    assert usage[0].output_tokens == 11
    # Sanity: the shape of the stream is otherwise unchanged.
    assert any(isinstance(e, TextDelta) for e in events)
    assert isinstance(events[-1], AssistantTurnEnd)


@pytest.mark.asyncio
async def test_openrouter_adapter_stamps_timings_via_inheritance():
    """OpenRouter subclasses OpenAI, so timing instrumentation inherits for
    free. Verify end-to-end rather than relying on the inheritance fact,
    since that's what matters when a user is debugging a slow OpenRouter
    turn."""
    from types import SimpleNamespace

    class FakeStream:
        def __init__(self):
            self._items = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="x", tool_calls=None),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                ),
            ]

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    clock = iter([10.0, 10.1, 10.5])

    class FakeCompletions:
        async def create(self, **_kw):
            return FakeStream()

    class FakeChat:
        completions = FakeCompletions()

    p = OpenRouterProvider(api_key="or")
    p._client.chat = FakeChat()

    with patch.object(_provider_mod.time, "monotonic", side_effect=lambda: next(clock)):
        events = [
            e async for e in p.stream_turn(
                model="x/y",
                system="",
                messages=[UserText(text="hi")],
                tools=[],
            )
        ]
    usage = next(e for e in events if isinstance(e, TurnUsageEvent))
    assert usage.ttft_ms == pytest.approx(100.0)
    assert usage.duration_ms == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# Runtime pipes timings into persisted TurnMetrics + forwarded TurnUsageEvent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wire frame + CLI rendering: the numbers reach the user, not just the DB
# ---------------------------------------------------------------------------


def test_event_to_wire_includes_timings_when_present():
    from ark.runtime import event_to_wire

    frame = event_to_wire(TurnUsageEvent(
        input_tokens=1, output_tokens=2, model="m",
        ttft_ms=42.0, duration_ms=180.0,
    ))
    assert frame["type"] == "turn_usage"
    assert frame["ttft_ms"] == 42.0
    assert frame["duration_ms"] == 180.0


def test_event_to_wire_omits_timings_when_absent():
    """A frame from a stubbed test provider or an adapter that didn't
    instrument shouldn't carry stray `ttft_ms: null` keys — keeps the wire
    shape clean for clients doing `if "ttft_ms" in frame`."""
    from ark.runtime import event_to_wire

    frame = event_to_wire(TurnUsageEvent(input_tokens=1, output_tokens=2))
    assert "ttft_ms" not in frame
    assert "duration_ms" not in frame


def test_cli_usage_line_renders_timings(capsys):
    """The dim per-turn indicator must include both TTFT and total duration
    so the user can eyeball whether `fast` actually came back faster."""
    from ark.cli import _Ui

    ui = _Ui()
    ui.is_tty = False  # bypass wiping
    ui.usage(
        input_tokens=100, output_tokens=50,
        context_window=10_000, model="x/y",
        ttft_ms=250.0, duration_ms=1200.0,
    )
    err = capsys.readouterr().err
    assert "ttft 250ms" in err
    assert "total 1.2s" in err


def test_cli_usage_line_skips_timing_section_when_absent(capsys):
    """Turns from adapters that don't yet instrument (or test stubs) must
    render the legacy indicator with no empty `· / ·` fragment."""
    from ark.cli import _Ui

    ui = _Ui()
    ui.is_tty = False
    ui.usage(
        input_tokens=100, output_tokens=50,
        context_window=10_000, model="x/y",
    )
    err = capsys.readouterr().err
    assert "ttft" not in err
    assert "total" not in err
    assert "100/10,000 ctx" in err  # legacy shape intact


@pytest.mark.asyncio
async def test_runtime_persists_adapter_timings(ark_home, tmp_path):
    """End-to-end: a stub provider yields a TurnUsageEvent with timings;
    the runtime must write those fields into the persisted TurnMetrics row
    AND re-emit them on the outbound TurnUsageEvent (so clients that
    render usage live see the same numbers as the history row)."""
    from ark import db, runtime
    from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
    from ark.types import AssistantTurnEnd, TextDelta

    class StubProvider:
        DEFAULT_SPEED_HINTS: dict = {}

        def __init__(self, **_kw):
            pass

        async def stream_turn(self, **_kw):
            yield TextDelta(text="ok")
            yield TurnUsageEvent(
                input_tokens=3, output_tokens=5, model="m",
                ttft_ms=12.0, duration_ms=250.0,
            )
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    conn = db.init_db()
    ws = tmp_path / "ws"; ws.mkdir()
    cfg = Config(
        server=ServerConfig(host="h", port=1, auth_secret="s"),
        providers={"p": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={"a": AgentConfig(name="a", provider="p", model="m", workspace=ws)},
    )
    sid = runtime.create_session(conn, "a", "conversational")
    agent = cfg.agents["a"]

    emitted: list = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=agent, session_id=sid,
        user_text="hi", provider_factory=lambda *a, **k: StubProvider(),
    ):
        emitted.append(evt)

    # The forwarded TurnUsageEvent carries the timings (same numbers as
    # the adapter supplied, not re-measured by the runtime).
    usage = [e for e in emitted if isinstance(e, TurnUsageEvent)]
    assert len(usage) == 1
    assert usage[0].ttft_ms == 12.0
    assert usage[0].duration_ms == 250.0
    # And so does the persisted TurnMetrics row.
    history = runtime.load_history(conn, sid)
    metrics = [m for m in history if isinstance(m, TurnMetrics)]
    assert len(metrics) == 1
    assert metrics[0].ttft_ms == 12.0
    assert metrics[0].duration_ms == 250.0
