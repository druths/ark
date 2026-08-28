"""Session compaction: CompactionSummary mechanism, proactive threshold,
reactive retry, event stream, and edge cases."""

from __future__ import annotations

import pytest

from ark import db, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    CompactionCompletedEvent,
    CompactionFailedEvent,
    CompactionSkippedEvent,
    CompactionStartedEvent,
    CompactionSummary,
    RunEnd,
    RunErrorEvent,
    SessionContext,
    TextDelta,
    ToolCallEvent,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(tmp_path, *, enabled=True, threshold=0.85, max_context=1000):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    agent = AgentConfig(
        name="scribe",
        provider="a",
        model="claude-sonnet-4-6",
        workspace=ws,
        compaction_enabled=enabled,
        compaction_threshold=threshold,
        max_context_tokens=max_context,
    )
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={"scribe": agent},
    )


class _ScriptedProvider:
    """Yields a queued script of provider events per call. Records what
    system prompt + message list was passed on each stream_turn."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096, prompt_caching=False):
        self.calls.append({"system": system, "messages": list(messages)})
        script = self._scripts.pop(0)
        for evt in script:
            if isinstance(evt, Exception):
                raise evt
            yield evt


def _factory(provider):
    return lambda *_a, **_k: provider


# ---------------------------------------------------------------------------
# CompactionSummary round-trip
# ---------------------------------------------------------------------------


def test_compaction_summary_round_trip():
    """The message survives the DB round-trip and is loadable as its type."""
    from ark.types import message_from_row, message_to_row

    original = CompactionSummary(text="prior summary", reason="auto:test")
    role, content = message_to_row(original)
    assert role == "compaction_summary"
    restored = message_from_row(role, content)
    assert isinstance(restored, CompactionSummary)
    assert restored.text == "prior summary"
    assert restored.reason == "auto:test"


# ---------------------------------------------------------------------------
# History slice + system-prompt fold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_slices_history_and_folds_into_system_prompt(ark_home, tmp_path):
    """After a CompactionSummary row exists, the LLM should see:
    - system prompt with the summary text folded in
    - message list starting AFTER the summary (older messages hidden)"""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="Q1: what's the weather"))
    runtime.append_message(conn, sid, AssistantText(text="A1: sunny"))
    runtime.append_message(
        conn, sid, CompactionSummary(text="User asked about weather; I said sunny.", reason="test")
    )
    # This one is post-compaction and SHOULD appear in the message list:
    runtime.append_message(conn, sid, UserText(text="Q2: and tomorrow?"))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="A2: rainy", stop_reason="end")]])
    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="Q3: friday?",
        provider_factory=_factory(provider),
    ):
        pass

    seen = provider.calls[0]
    # Summary is in the system prompt:
    assert "Prior conversation (summarized" in seen["system"]
    assert "asked about weather" in seen["system"]
    # Message list contains ONLY Q2 (post-compaction) + Q3 (this turn) —
    # Q1 and A1 are hidden by the slice.
    kinds = [type(m).__name__ for m in seen["messages"]]
    texts = [getattr(m, "text", "") for m in seen["messages"]]
    assert "Q1" not in " ".join(texts)
    assert "A1" not in " ".join(texts)
    assert any("Q2" in t for t in texts)
    assert any("Q3" in t for t in texts)


@pytest.mark.asyncio
async def test_compaction_summary_row_never_leaks_to_llm(ark_home, tmp_path):
    """The CompactionSummary row itself must never appear in the message list
    sent to the provider (it goes into the system prompt instead)."""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="hi"))
    runtime.append_message(conn, sid, CompactionSummary(text="prev summary", reason="t"))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi again",
        provider_factory=_factory(provider),
    ):
        pass

    for m in provider.calls[0]["messages"]:
        assert not isinstance(m, CompactionSummary)


@pytest.mark.asyncio
async def test_latest_compaction_wins_when_multiple_exist(ark_home, tmp_path):
    """With multiple CompactionSummary rows, the runtime uses the latest one
    as the slice point and system-prompt fold."""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="ancient"))
    runtime.append_message(conn, sid, CompactionSummary(text="OLD summary", reason="t1"))
    runtime.append_message(conn, sid, UserText(text="middle"))
    runtime.append_message(conn, sid, CompactionSummary(text="NEW summary", reason="t2"))
    runtime.append_message(conn, sid, UserText(text="recent"))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="now",
        provider_factory=_factory(provider),
    ):
        pass

    seen = provider.calls[0]
    assert "NEW summary" in seen["system"]
    assert "OLD summary" not in seen["system"]
    texts = [getattr(m, "text", "") for m in seen["messages"]]
    joined = " ".join(texts)
    assert "ancient" not in joined
    assert "middle" not in joined
    assert "recent" in joined
    assert "now" in joined


# ---------------------------------------------------------------------------
# Proactive compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proactive_compaction_triggers_above_threshold(ark_home, tmp_path):
    """When last TurnMetrics.input_tokens exceeds threshold * context_window,
    a compaction runs BEFORE the user message is persisted."""
    cfg = _cfg(tmp_path, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    # Enough post-nothing history to clear the "at least 6 messages" floor.
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"user {i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"reply {i}"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=800, output_tokens=50))

    # Two provider calls: (1) summarizer, (2) the actual user turn.
    provider = _ScriptedProvider([
        [TextDelta(text="[SUMMARY of prior conversation]"),
         AssistantTurnEnd(text="[SUMMARY of prior conversation]", stop_reason="end")],
        [AssistantTurnEnd(text="answer", stop_reason="end")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="next question",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    # Compaction lifecycle events fired:
    started = [e for e in events if isinstance(e, CompactionStartedEvent)]
    completed = [e for e in events if isinstance(e, CompactionCompletedEvent)]
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].reason.startswith("auto:threshold")
    assert "800" in started[0].reason  # observed input_tokens in the reason
    assert completed[0].summary == "[SUMMARY of prior conversation]"

    # A CompactionSummary row was persisted:
    history = runtime.load_history(conn, sid)
    summaries = [m for m in history if isinstance(m, CompactionSummary)]
    assert len(summaries) == 1

    # The actual turn saw the summary in the system prompt and only the
    # new user message in the message list (the compaction sliced out
    # everything prior).
    actual_turn = provider.calls[1]
    assert "SUMMARY of prior conversation" in actual_turn["system"]
    turn_texts = [getattr(m, "text", "") for m in actual_turn["messages"]]
    assert turn_texts == ["next question"]


@pytest.mark.asyncio
async def test_proactive_compaction_skipped_below_threshold(ark_home, tmp_path):
    cfg = _cfg(tmp_path, threshold=0.85, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=500, output_tokens=10))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    assert not any(isinstance(e, CompactionStartedEvent) for e in events)


@pytest.mark.asyncio
async def test_proactive_skipped_when_metrics_stale_post_compaction(ark_home, tmp_path):
    """A CompactionSummary AFTER the latest TurnMetrics means the metrics
    reflect pre-compaction fill — stale. Skip to avoid a false trigger."""
    cfg = _cfg(tmp_path, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=900, output_tokens=50))
    # Compaction ran AFTER those metrics were recorded:
    runtime.append_message(conn, sid, CompactionSummary(text="summarized", reason="t"))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    assert not any(isinstance(e, CompactionStartedEvent) for e in events)


@pytest.mark.asyncio
async def test_proactive_skipped_with_too_little_history_since_compaction(ark_home, tmp_path):
    """No point compacting a handful of messages — the floor guards
    pathological loops."""
    cfg = _cfg(tmp_path, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, CompactionSummary(text="prev", reason="t"))
    # Only 2 post-compaction messages — under the floor of 6.
    runtime.append_message(conn, sid, UserText(text="q"))
    runtime.append_message(conn, sid, AssistantText(text="a"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=900, output_tokens=50))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="q2",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    assert not any(isinstance(e, CompactionStartedEvent) for e in events)


@pytest.mark.asyncio
async def test_proactive_skipped_first_turn(ark_home, tmp_path):
    """No TurnMetrics → nothing to threshold against → skip."""
    cfg = _cfg(tmp_path, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="first",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    assert not any(isinstance(e, CompactionStartedEvent) for e in events)


@pytest.mark.asyncio
async def test_compaction_skipped_event_when_disabled(ark_home, tmp_path):
    """With compaction disabled, threshold-crossing emits compaction_skipped
    so the client can warn — silent inaction on a user-visible risk is bad UX."""
    cfg = _cfg(tmp_path, enabled=False, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=900, output_tokens=50))

    provider = _ScriptedProvider([[AssistantTurnEnd(text="ok", stop_reason="end")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="hi",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    skipped = [e for e in events if isinstance(e, CompactionSkippedEvent)]
    assert len(skipped) == 1
    assert "disabled" in skipped[0].reason
    # No CompactionSummary row was written.
    history = runtime.load_history(conn, sid)
    assert not any(isinstance(m, CompactionSummary) for m in history)


# ---------------------------------------------------------------------------
# Reactive compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reactive_compaction_on_context_too_long_retries(ark_home, tmp_path):
    """When the provider raises context_too_long on a fresh user turn, the
    runtime compacts and retries the same iteration once."""
    cfg = _cfg(tmp_path, threshold=0.99, max_context=1000)  # threshold above what's set
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))

    # 3 scripted calls: (1) first turn attempt raises context_too_long,
    # (2) summarizer produces summary text, (3) retry succeeds.
    ctx_err = Exception("prompt is too long")
    provider = _ScriptedProvider([
        [ctx_err],
        [TextDelta(text="SUMMARY"), AssistantTurnEnd(text="SUMMARY", stop_reason="end")],
        [TextDelta(text="retry answer"),
         AssistantTurnEnd(text="retry answer", stop_reason="end")],
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="my question",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    # Compaction ran with the reactive reason.
    started = [e for e in events if isinstance(e, CompactionStartedEvent)]
    completed = [e for e in events if isinstance(e, CompactionCompletedEvent)]
    assert len(started) == 1
    assert started[0].reason == "reactive:context_too_long"
    assert len(completed) == 1

    # No RunError was persisted — the retry succeeded.
    history = runtime.load_history(conn, sid)
    from ark.types import RunError as _RE
    assert not any(isinstance(m, _RE) for m in history)

    # Retry call saw the summary in system prompt and just the user message
    # in the message list.
    retry_call = provider.calls[2]
    assert "SUMMARY" in retry_call["system"]
    texts = [getattr(m, "text", "") for m in retry_call["messages"]]
    assert texts == ["my question"]

    # And the user got an assistant turn:
    ends = [e for e in events if isinstance(e, AssistantTurnEnd)]
    assert ends[-1].text == "retry answer"


@pytest.mark.asyncio
async def test_reactive_compaction_not_used_twice_in_same_turn(ark_home, tmp_path):
    """If context_too_long fires a second time after a successful compaction,
    the runtime does NOT compact again — falls through to the error path."""
    cfg = _cfg(tmp_path, threshold=0.99, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))

    ctx_err = Exception("prompt is too long")
    provider = _ScriptedProvider([
        [ctx_err],                                                        # first fail
        [TextDelta(text="S"), AssistantTurnEnd(text="S", stop_reason="end")],  # compaction
        [ctx_err],                                                        # second fail
    ])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="q",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    started = [e for e in events if isinstance(e, CompactionStartedEvent)]
    assert len(started) == 1  # only one compaction attempt total
    error_events = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(error_events) == 1
    assert error_events[0].code == "context_too_long"


@pytest.mark.asyncio
async def test_reactive_compaction_failure_falls_through_to_error(ark_home, tmp_path):
    """If the summarizer itself raises, emit compaction_failed and then
    proceed with the normal error path — the client sees both signals."""
    cfg = _cfg(tmp_path, threshold=0.99, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))

    ctx_err = Exception("input token count exceeds")
    summarizer_err = Exception("summarizer network flake")
    provider = _ScriptedProvider([[ctx_err], [summarizer_err]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="q",
        provider_factory=_factory(provider),
    ):
        events.append(evt)

    failed = [e for e in events if isinstance(e, CompactionFailedEvent)]
    error_events = [e for e in events if isinstance(e, RunErrorEvent)]
    assert len(failed) == 1
    assert len(error_events) == 1
    assert error_events[0].code == "context_too_long"


@pytest.mark.asyncio
async def test_reactive_compaction_disabled_falls_through(ark_home, tmp_path):
    """With compaction_enabled=False, a context_too_long error goes straight
    to the existing error path — no compaction attempted."""
    cfg = _cfg(tmp_path, enabled=False, threshold=0.99, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))

    provider = _ScriptedProvider([[Exception("prompt is too long")]])
    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="q",
        provider_factory=_factory(provider),
    ):
        events.append(evt)
    assert not any(isinstance(e, CompactionStartedEvent) for e in events)
    assert any(isinstance(e, RunErrorEvent) for e in events)


# ---------------------------------------------------------------------------
# Prior summary preservation across successive compactions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_compaction_includes_prior_summary_in_summarizer_prompt(ark_home, tmp_path):
    """When compaction runs a second time, the prior CompactionSummary text
    should appear in the summarizer's system prompt so info isn't lost."""
    cfg = _cfg(tmp_path, threshold=0.5, max_context=1000)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, CompactionSummary(text="PRIOR SUMMARY BODY", reason="t1"))
    for i in range(6):
        runtime.append_message(conn, sid, UserText(text=f"u{i}"))
        runtime.append_message(conn, sid, AssistantText(text=f"a{i}"))
    runtime.append_message(conn, sid, TurnMetrics(input_tokens=900, output_tokens=50))

    provider = _ScriptedProvider([
        [TextDelta(text="NEW"), AssistantTurnEnd(text="NEW", stop_reason="end")],  # summarizer
        [AssistantTurnEnd(text="answer", stop_reason="end")],                         # actual turn
    ])
    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="q",
        provider_factory=_factory(provider),
    ):
        pass

    summarizer_call = provider.calls[0]
    assert "PRIOR SUMMARY BODY" in summarizer_call["system"]


# ---------------------------------------------------------------------------
# Wire format for the new events
# ---------------------------------------------------------------------------


def test_wire_format_for_compaction_events():
    from ark.runtime import event_to_wire

    assert event_to_wire(
        CompactionStartedEvent(reason="auto:x", input_tokens=800,
                               context_window=1000, model="claude-x")
    ) == {
        "type": "compaction_started",
        "reason": "auto:x",
        "input_tokens": 800,
        "context_window": 1000,
        "model": "claude-x",
    }
    assert event_to_wire(
        CompactionCompletedEvent(summary="s", reason="auto:x")
    ) == {"type": "compaction_completed", "summary": "s", "reason": "auto:x"}
    assert event_to_wire(
        CompactionFailedEvent(code="rate_limit", message="429", reason="reactive:x")
    ) == {"type": "compaction_failed", "code": "rate_limit", "message": "429",
          "reason": "reactive:x"}
    assert event_to_wire(
        CompactionSkippedEvent(reason="disabled:x", input_tokens=900,
                               context_window=1000)
    ) == {"type": "compaction_skipped", "reason": "disabled:x",
          "input_tokens": 900, "context_window": 1000}
