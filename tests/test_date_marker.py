"""DateMarker — "time has passed" cue in long-running sessions.

Covers:
- Round-trip of the message kind
- Env stanza always shows today's UTC date
- Insertion check: fires only on calendar-date change, only for
  conversational sessions, only when a previous UserText exists
- _rewrite_for_llm substitutes DateMarker → synthetic UserText notification
- Live WS wire shape (`date_marker`) carries `event_id`
- Marker rows don't leak into /history as raw UserText
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from ark import db, runtime
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.types import (
    AssistantTurnEnd,
    DateMarker,
    DateMarkerEvent,
    TextDelta,
    TurnUsageEvent,
    UserText,
    message_from_row,
    message_to_row,
)


def _cfg(tmp_path):
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


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_date_marker_round_trip():
    m = DateMarker(from_date="2026-09-29", to_date="2026-10-05", elapsed_days=6)
    role, content = message_to_row(m)
    assert role == "date_marker"
    restored = message_from_row(role, content)
    assert isinstance(restored, DateMarker)
    assert restored.from_date == "2026-09-29"
    assert restored.to_date == "2026-10-05"
    assert restored.elapsed_days == 6


# ---------------------------------------------------------------------------
# Env stanza: today's UTC date
# ---------------------------------------------------------------------------


def test_system_prompt_includes_todays_utc_date(tmp_path, ark_home):
    cfg = _cfg(tmp_path)
    prompt = runtime.system_prompt(cfg.agents["scribe"])
    today = datetime.now(timezone.utc).date().isoformat()
    assert today in prompt
    assert "Today's date (UTC)" in prompt


# ---------------------------------------------------------------------------
# Insertion check
# ---------------------------------------------------------------------------


def _backdate_user(conn, session_id, ms_ago: int) -> None:
    """Rewrite the last UserText row's created_at to be `ms_ago` ms before
    `now_ms()`. Lets us simulate the user coming back after an interval."""
    now = runtime.now_ms()
    conn.execute(
        "UPDATE messages SET created_at = ? "
        "WHERE session_id = ? AND role = 'user' AND seq = ("
        "SELECT MAX(seq) FROM messages WHERE session_id = ? AND role = 'user')",
        (now - ms_ago, session_id, session_id),
    )


def test_marker_inserted_when_date_changes(ark_home, tmp_path):
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="yesterday's question"))
    # Backdate by ~26 hours so the UTC date is definitively yesterday
    # (regardless of what time of day the test runs).
    _backdate_user(conn, sid, 26 * 60 * 60 * 1000)

    evt = runtime._maybe_insert_date_marker(conn, sid)
    assert evt is not None
    assert evt.elapsed_days >= 1
    assert evt.to_date == datetime.now(timezone.utc).date().isoformat()

    # Marker is persisted.
    markers = [m for m in runtime.load_history(conn, sid) if isinstance(m, DateMarker)]
    assert len(markers) == 1


def test_marker_not_inserted_when_same_day(ark_home, tmp_path):
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="just now"))

    evt = runtime._maybe_insert_date_marker(conn, sid)
    assert evt is None
    assert not any(
        isinstance(m, DateMarker) for m in runtime.load_history(conn, sid)
    )


def test_marker_not_inserted_on_first_turn(ark_home, tmp_path):
    """No previous UserText → nothing to compare against → no marker."""
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    evt = runtime._maybe_insert_date_marker(conn, sid)
    assert evt is None


def test_marker_skipped_for_cron_sessions(ark_home, tmp_path):
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", kind="cron", cron_id="daily")
    runtime.append_message(conn, sid, UserText(text="prior cron prompt"))
    _backdate_user(conn, sid, 48 * 60 * 60 * 1000)
    assert runtime._maybe_insert_date_marker(conn, sid) is None


def test_marker_skipped_for_heartbeat_sessions(ark_home, tmp_path):
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", kind="heartbeat")
    runtime.append_message(conn, sid, UserText(text="heartbeat prompt"))
    _backdate_user(conn, sid, 48 * 60 * 60 * 1000)
    assert runtime._maybe_insert_date_marker(conn, sid) is None


def test_marker_ignores_non_user_messages_for_comparison(ark_home, tmp_path):
    """`post_to_session` activity (which writes AssistantText with
    injected_from) between user turns must not count as "user came back."
    The signal is specifically the previous USER turn's date."""
    from ark.types import AssistantText

    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="original turn"))
    _backdate_user(conn, sid, 48 * 60 * 60 * 1000)
    # Simulate a cron injecting a message between user turns (same day, recent).
    runtime.append_message(
        conn, sid, AssistantText(text="injected briefing", injected_from="other-sid")
    )

    # The marker check should still fire because the previous USER text
    # was 2 days ago — the AssistantText row doesn't count.
    evt = runtime._maybe_insert_date_marker(conn, sid)
    assert evt is not None
    assert evt.elapsed_days >= 1


# ---------------------------------------------------------------------------
# _rewrite_for_llm substitution
# ---------------------------------------------------------------------------


def test_rewrite_substitutes_date_marker_as_synthetic_user_text():
    msgs = [
        UserText(text="hi"),
        DateMarker(from_date="2026-09-29", to_date="2026-10-05", elapsed_days=6),
        UserText(text="back"),
    ]
    out = runtime._rewrite_for_llm(msgs)
    # DateMarker is substituted; original row is replaced in the LLM list.
    kinds = [type(m).__name__ for m in out]
    assert "DateMarker" not in kinds
    assert kinds.count("UserText") == 3

    notification = out[1]
    assert isinstance(notification, UserText)
    assert "2026-10-05" in notification.text
    assert "2026-09-29" in notification.text
    assert "6 days ago" in notification.text
    assert "recalibrate" in notification.text


def test_date_marker_notification_singular_day():
    msg = DateMarker(from_date="2026-10-04", to_date="2026-10-05", elapsed_days=1)
    text = runtime._date_marker_notification(msg)
    assert "1 day ago" in text


# ---------------------------------------------------------------------------
# event_to_wire
# ---------------------------------------------------------------------------


def test_date_marker_event_to_wire():
    w = runtime.event_to_wire(DateMarkerEvent(
        from_date="2026-09-29", to_date="2026-10-05", elapsed_days=6, row_id=42
    ))
    assert w["type"] == "date_marker"
    assert w["from_date"] == "2026-09-29"
    assert w["to_date"] == "2026-10-05"
    assert w["elapsed_days"] == 6
    assert w["event_id"] == 42


def test_date_marker_event_without_row_id_omits_event_id():
    w = runtime.event_to_wire(DateMarkerEvent(
        from_date="2026-09-29", to_date="2026-10-05", elapsed_days=6
    ))
    assert "event_id" not in w


# ---------------------------------------------------------------------------
# End-to-end: run_user_turn yields DateMarkerEvent + model sees notification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_user_turn_yields_date_marker_event(ark_home, tmp_path):
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="old turn"))
    _backdate_user(conn, sid, 48 * 60 * 60 * 1000)

    captured: dict = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            captured["system"] = system
            captured["messages"] = list(messages)
            yield TurnUsageEvent(input_tokens=50, output_tokens=10, model=model)
            yield TextDelta(text="hi")
            yield AssistantTurnEnd(text="hi", stop_reason="end_turn")

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="I'm back",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        events.append(evt)

    # DateMarkerEvent appears in the yielded stream, with a real row_id.
    markers = [e for e in events if isinstance(e, DateMarkerEvent)]
    assert len(markers) == 1
    assert markers[0].row_id is not None
    assert markers[0].elapsed_days >= 1

    # The marker's row is in the history.
    history = runtime.load_history(conn, sid)
    assert any(isinstance(m, DateMarker) for m in history)

    # The model's message list contains the synthetic UserText notification
    # (not the raw DateMarker). The provider never sees the DateMarker row.
    for m in captured["messages"]:
        assert not isinstance(m, DateMarker)
    notifications = [
        m.text for m in captured["messages"]
        if isinstance(m, UserText) and "system notification" in m.text
    ]
    assert len(notifications) == 1
    assert "recalibrate" in notifications[0]

    # The system prompt still shows today's UTC date.
    today = datetime.now(timezone.utc).date().isoformat()
    assert today in captured["system"]


@pytest.mark.asyncio
async def test_run_user_turn_no_marker_on_same_day(ark_home, tmp_path):
    """Baseline: back-to-back user turns on the same UTC day emit no marker."""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="first"))

    class _StubProvider:
        async def stream_turn(self, **_kw):
            yield TurnUsageEvent(input_tokens=50, output_tokens=10, model="m")
            yield TextDelta(text="ok")
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="second",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        events.append(evt)
    assert not any(isinstance(e, DateMarkerEvent) for e in events)
    assert not any(
        isinstance(m, DateMarker) for m in runtime.load_history(conn, sid)
    )


# ---------------------------------------------------------------------------
# Client-supplied timezone
# ---------------------------------------------------------------------------


def test_coerce_tz_fallbacks_to_utc():
    """None, empty, and garbage strings all fall through to UTC."""
    assert runtime._coerce_tz(None) == "UTC"
    assert runtime._coerce_tz("") == "UTC"
    assert runtime._coerce_tz("Not/A/Zone") == "UTC"


def test_coerce_tz_accepts_valid_iana():
    assert runtime._coerce_tz("America/Los_Angeles") == "America/Los_Angeles"
    assert runtime._coerce_tz("Europe/London") == "Europe/London"
    assert runtime._coerce_tz("UTC") == "UTC"


def test_marker_uses_client_timezone_when_supplied(ark_home, tmp_path):
    """When a client passes a timezone, the comparison uses THAT zone, not
    UTC. The marker's `timezone` field records which zone was used."""
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="earlier"))
    # Backdate by ~30 hours — guaranteed yesterday in any zone.
    _backdate_user(conn, sid, 30 * 60 * 60 * 1000)

    evt = runtime._maybe_insert_date_marker(
        conn, sid, client_tz="America/Los_Angeles"
    )
    assert evt is not None
    assert evt.timezone == "America/Los_Angeles"

    from datetime import datetime
    from zoneinfo import ZoneInfo
    expected_today = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    assert evt.to_date == expected_today


def test_marker_tz_persisted_on_row(ark_home, tmp_path):
    """The DateMarker row records the TZ it was computed in."""
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="hi"))
    _backdate_user(conn, sid, 30 * 60 * 60 * 1000)
    runtime._maybe_insert_date_marker(conn, sid, client_tz="America/Los_Angeles")
    history = runtime.load_history(conn, sid)
    markers = [m for m in history if isinstance(m, DateMarker)]
    assert markers[0].timezone == "America/Los_Angeles"


def test_marker_notification_includes_tz_name():
    msg = DateMarker(
        from_date="2026-10-03", to_date="2026-10-05", elapsed_days=2,
        timezone="America/Los_Angeles",
    )
    text = runtime._date_marker_notification(msg)
    assert "America/Los_Angeles" in text
    assert "UTC" not in text  # TZ is explicit, no UTC confusion


def test_marker_notification_defaults_to_utc_when_tz_unset():
    """Backwards-compat: markers written before this feature have
    timezone='UTC' by default (dataclass default). Notification reads
    cleanly."""
    msg = DateMarker(
        from_date="2026-10-03", to_date="2026-10-05", elapsed_days=2
    )  # no timezone kwarg → default "UTC"
    text = runtime._date_marker_notification(msg)
    assert "(UTC)" in text


def test_invalid_tz_falls_back_to_utc(ark_home, tmp_path):
    """Garbage TZ string → UTC fallback, no error raised. Marker still
    fires if the date actually changed in UTC."""
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="hi"))
    _backdate_user(conn, sid, 48 * 60 * 60 * 1000)
    evt = runtime._maybe_insert_date_marker(conn, sid, client_tz="Not/Real")
    assert evt is not None
    assert evt.timezone == "UTC"


def test_env_stanza_shows_local_date_when_tz_supplied(tmp_path, ark_home):
    """Env stanza shows '(America/Los_Angeles): <local date>  (UTC: <utc date>)'
    when a client timezone is passed in."""
    cfg = _cfg(tmp_path)
    prompt = runtime.system_prompt(
        cfg.agents["scribe"], client_timezone="America/Los_Angeles"
    )
    assert "America/Los_Angeles" in prompt
    # The UTC date should also appear (both are shown when a TZ is given).
    from datetime import datetime, timezone
    today_utc = datetime.now(timezone.utc).date().isoformat()
    assert today_utc in prompt


def test_env_stanza_utc_only_when_tz_absent(tmp_path, ark_home):
    """Backwards-compat: no client_timezone → the old 'Today's date (UTC): X'
    line, unchanged."""
    cfg = _cfg(tmp_path)
    prompt = runtime.system_prompt(cfg.agents["scribe"])
    assert "Today's date (UTC):" in prompt
    # No TZ name other than UTC should appear in the stanza.
    for forbidden in ("America/Los_Angeles", "Europe/London"):
        assert forbidden not in prompt


@pytest.mark.asyncio
async def test_run_user_turn_threads_tz_through(ark_home, tmp_path):
    """End-to-end: a user_message with timezone="America/Los_Angeles" lands
    with the marker + env stanza + synthetic notification all in that TZ."""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")
    runtime.append_message(conn, sid, UserText(text="earlier"))
    _backdate_user(conn, sid, 30 * 60 * 60 * 1000)

    captured: dict = {}

    class _StubProvider:
        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            captured["system"] = system
            captured["messages"] = list(messages)
            yield TurnUsageEvent(input_tokens=50, output_tokens=10, model=model)
            yield TextDelta(text="ok")
            yield AssistantTurnEnd(text="ok", stop_reason="end_turn")

    events = []
    async for evt in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="back from LA",
        provider_factory=lambda *_a, **_k: _StubProvider(),
        client_timezone="America/Los_Angeles",
    ):
        events.append(evt)

    marker = next(e for e in events if isinstance(e, DateMarkerEvent))
    assert marker.timezone == "America/Los_Angeles"

    # Env stanza shows the LA date, not just UTC.
    assert "America/Los_Angeles" in captured["system"]

    # Model sees the TZ in the synthetic notification.
    notif = next(
        m for m in captured["messages"]
        if isinstance(m, UserText) and "system notification" in m.text
    )
    assert "America/Los_Angeles" in notif.text


def test_marker_event_to_wire_includes_timezone():
    w = runtime.event_to_wire(DateMarkerEvent(
        from_date="2026-10-03", to_date="2026-10-05", elapsed_days=2,
        timezone="America/Los_Angeles", row_id=99,
    ))
    assert w["timezone"] == "America/Los_Angeles"
    assert w["event_id"] == 99


# ---------------------------------------------------------------------------
# WS wire: `user_message` frame carries `timezone` field through
# ---------------------------------------------------------------------------


def test_server_ws_forwards_timezone_to_runtime(ark_home, tmp_path, monkeypatch):
    """A `user_message` with `timezone` lands as `client_timezone` on the
    runtime call."""
    import json as _json
    from fastapi.testclient import TestClient
    from ark.server import create_app
    from ark.config import ProviderConfig, ServerConfig

    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="claude-sonnet-4-6",
                workspace=tmp_path / "ws",
            )
        },
    )
    (tmp_path / "ws").mkdir(exist_ok=True)

    captured_kwargs: dict = {}

    async def _fake_run_and_publish(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(runtime, "run_and_publish", _fake_run_and_publish)

    client = TestClient(create_app(cfg))
    sid = client.post(
        "/agents/scribe/sessions", headers={"Authorization": "Bearer x"}
    ).json()["id"]

    with client.websocket_connect("/events?token=x") as ws:
        ws.send_text(_json.dumps({
            "type": "user_message",
            "session_id": sid,
            "text": "hi",
            "timezone": "America/Los_Angeles",
        }))
        # The task is spawned via create_task; give it a tick to execute.
        import time
        for _ in range(50):
            if "client_timezone" in captured_kwargs:
                break
            time.sleep(0.02)

    assert captured_kwargs.get("client_timezone") == "America/Los_Angeles"


def test_server_ws_non_string_timezone_falls_back_to_none(ark_home, tmp_path, monkeypatch):
    """A `timezone: 42` (wrong type) is treated as absent — the runtime
    gets None and falls back to UTC. Doesn't reject the turn."""
    import json as _json
    from fastapi.testclient import TestClient
    from ark.server import create_app
    from ark.config import ProviderConfig, ServerConfig

    cfg = Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="claude-sonnet-4-6",
                workspace=tmp_path / "ws",
            )
        },
    )
    (tmp_path / "ws").mkdir(exist_ok=True)

    captured_kwargs: dict = {}

    async def _fake_run_and_publish(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(runtime, "run_and_publish", _fake_run_and_publish)

    client = TestClient(create_app(cfg))
    sid = client.post(
        "/agents/scribe/sessions", headers={"Authorization": "Bearer x"}
    ).json()["id"]

    with client.websocket_connect("/events?token=x") as ws:
        ws.send_text(_json.dumps({
            "type": "user_message",
            "session_id": sid,
            "text": "hi",
            "timezone": 42,
        }))
        import time
        for _ in range(50):
            if "client_timezone" in captured_kwargs:
                break
            time.sleep(0.02)

    assert captured_kwargs.get("client_timezone") is None


# ---------------------------------------------------------------------------
# CLI local-TZ detection
# ---------------------------------------------------------------------------


def test_cli_detect_local_timezone_returns_sensible_value_or_none():
    """On a dev macOS box, /etc/localtime typically points to a zoneinfo
    file. On stripped-down CI images, we may return None (fine — server
    falls back to UTC)."""
    from ark.cli import _detect_local_timezone
    tz = _detect_local_timezone()
    if tz is not None:
        # Must be a parseable IANA zone when present.
        from zoneinfo import ZoneInfo
        ZoneInfo(tz)
