"""Session runtime: persistence, turn loop, tool dispatch."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
import uuid
from typing import AsyncIterator

from dataclasses import asdict, is_dataclass

from . import broker, models, paths, projects, tools
from .config import AgentConfig, Config
from .provider import AnthropicProvider, Provider
from .types import (
    AssistantText,
    AssistantTurnEnd,
    CompactionCompletedEvent,
    CompactionFailedEvent,
    CompactionSkippedEvent,
    CompactionStartedEvent,
    CompactionSummary,
    DateMarker,
    DateMarkerEvent,
    Message,
    Project,
    ProjectAssignmentChanged,
    RunEnd,
    RunError,
    RunErrorEvent,
    RuntimeEvent,
    SessionContext,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    TurnMetrics,
    TurnUsageEvent,
    UserText,
    message_from_row,
    message_to_row,
)


# Default per-turn cumulative token budget (input + output summed across
# iterations of the model→tools loop). Replaces the old hardcoded
# max_iterations=16 cap. Generous by design — ordinary turns run
# 5–25k total; a runaway hitting 500k is almost certainly broken. Overrides:
# AgentConfig.max_turn_tokens (per-agent) and crons.max_tokens (per-cron).
DEFAULT_TURN_TOKEN_BUDGET = 500_000


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


def make_provider(provider_type: str, *, api_key: str, base_url: str | None = None) -> Provider:
    if provider_type == "anthropic":
        return AnthropicProvider(api_key=api_key, base_url=base_url)
    if provider_type == "openai":
        from .provider import OpenAIProvider

        return OpenAIProvider(api_key=api_key, base_url=base_url)
    if provider_type == "openrouter":
        from .provider import OpenRouterProvider

        return OpenRouterProvider(api_key=api_key, base_url=base_url)
    if provider_type == "google":
        from .provider import GoogleProvider

        return GoogleProvider(api_key=api_key, base_url=base_url)
    raise ValueError(f"unsupported provider_type: {provider_type}")


# ---------------------------------------------------------------------------
# Per-session in-memory state (e.g. loaded skills)
# ---------------------------------------------------------------------------


_session_loaded_skills: dict[str, set[str]] = {}


def loaded_skills(session_id: str) -> set[str]:
    return _session_loaded_skills.setdefault(session_id, set())


def reset_session_state(session_id: str) -> None:
    _session_loaded_skills.pop(session_id, None)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


def now_ms() -> int:
    return int(time.time() * 1000)


def create_session(
    conn: sqlite3.Connection,
    agent_name: str,
    kind: str = "conversational",
    project_id: str | None = None,
    cron_id: str | None = None,
    metadata: dict | None = None,
) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO sessions(id, agent_name, kind, created_at, project_id, cron_id, "
        "metadata_json) VALUES (?,?,?,?,?,?,?)",
        (
            sid,
            agent_name,
            kind,
            now_ms(),
            project_id,
            cron_id,
            json.dumps(metadata) if metadata else None,
        ),
    )
    return sid


def session_metadata(conn: sqlite3.Connection, session_id: str) -> dict:
    """Client-supplied session metadata ({} when none). Server-side only:
    surfaced to skills via ToolContext, never to the model."""

    row = conn.execute(
        "SELECT metadata_json FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None or not row["metadata_json"]:
        return {}
    try:
        data = json.loads(row["metadata_json"])
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def session_project(conn: sqlite3.Connection, session_id: str) -> Project | None:
    """Return the Project bound to a session, or None if the session is
    project-less (or the project has been deleted)."""

    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None or row["project_id"] is None:
        return None
    p = projects.get(conn, row["project_id"])
    if p is None or p.deleted_at is not None:
        return None
    return p


def list_sessions(
    conn: sqlite3.Connection,
    agent_name: str,
    *,
    kind: str | None = None,
    limit: int = 50,
) -> list[dict]:
    sql = "SELECT id, agent_name, kind, created_at, ended_at FROM sessions WHERE agent_name = ?"
    params: list = [agent_name]
    if kind:
        sql += " AND kind = ?"
        params.append(kind)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def session_exists(conn: sqlite3.Connection, session_id: str, agent_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE id = ? AND agent_name = ?",
        (session_id, agent_name),
    ).fetchone()
    return row is not None


def delete_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    reset_session_state(session_id)


def load_history(conn: sqlite3.Connection, session_id: str) -> list[Message]:
    rows = conn.execute(
        "SELECT role, content_json FROM messages WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    return [message_from_row(r["role"], json.loads(r["content_json"])) for r in rows]


def append_message(conn: sqlite3.Connection, session_id: str, msg: Message) -> int:
    """Append `msg` to the session's message log and return the new row's
    globally-monotonic `id` (autoincrement on `messages.id`).

    That id is the same one `GET /events` exposes as `next_since_id`, so
    live events published via the broker can carry the same identifier
    downstream clients see on catch-up — enabling deterministic dedupe
    across the live-WS and catch-up-REST surfaces (see docs/sessions.md's
    "Event ids" section)."""
    role, content = message_to_row(msg)
    next_seq = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    cur = conn.execute(
        "INSERT INTO messages(session_id, seq, role, content_json, created_at) "
        "VALUES (?,?,?,?,?)",
        (session_id, next_seq, role, json.dumps(content), now_ms()),
    )
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def system_prompt(
    agent: AgentConfig,
    contexts: list[SessionContext] | None = None,
    project: Project | None = None,
    compaction_summary: str | None = None,
    client_timezone: str | None = None,
) -> str:
    """Build the system prompt.

    `client_timezone` is the IANA zone name the current turn's client
    supplied (or None/UTC). The Environment stanza's "today's date" line
    is rendered in that zone, with UTC also shown parenthetically when
    the two differ.

    Layers (top to bottom):
      1. The user's `session_context.md` — agent identity / persona
      2. The Environment stanza — runtime facts (workspace path, available
         file/shell/upload helpers, today's date)
      3. Project framing — only when this session is bound to a project
      4. Any client-supplied SessionContext messages, concatenated in order
      5. Prior-conversation summary — only after a compaction has occurred
    """

    ctx_path = paths.agent_dir(agent.name) / "session_context.md"
    body = (
        ctx_path.read_text()
        if ctx_path.exists()
        else f"You are {agent.name}, an agent in the Ark harness."
    )
    from datetime import datetime, timezone as _tz
    today_utc = datetime.now(_tz.utc).date().isoformat()
    tz_label = _coerce_tz(client_timezone)
    if tz_label == "UTC":
        date_line = (
            f"- Today's date (UTC): {today_utc}. Your training data has a cutoff; "
            "treat this field as the ground truth for \"today\" and defer to "
            "`get_current_time` when you need the exact wall-clock time.\n"
        )
    else:
        local_date = _now_local_date(tz_label)
        date_line = (
            f"- Today's date ({tz_label}): {local_date}  (UTC: {today_utc}). "
            "Your training data has a cutoff; treat this field as the ground "
            "truth for \"today\" and defer to `get_current_time` (returns UTC) "
            "when you need the exact wall-clock time.\n"
        )
    env = (
        "\n\n---\n"
        "Environment (managed by the Ark harness, do not invent paths):\n"
        f"- Your name: {agent.name}\n"
        f"- Your workspace directory: {agent.workspace}\n"
        f"{date_line}"
        "- File and shell tools (read_file, write_file, list_files, run_command) "
        "operate on real paths on this server. The current working directory for "
        "each tool call is your workspace above. When in doubt about where a file "
        "lives, call list_files first instead of guessing.\n"
        "- Files the user attaches arrive in `uploads/` (relative to your workspace, "
        "or to the project root if this session is in a project — see below). "
        "Newer uploads of the same name are auto-suffixed (e.g. `report-2.pdf`). "
        "Use `list_uploads` to see what's available, newest first.\n"
        "- To hand a file back to the user, write it anywhere in your workspace "
        "and then call `share_with_client(path)`. The user's client will be "
        "notified and given a download link.\n"
    )
    out = body + env
    if project is not None:
        proj = (
            "\n\n---\n"
            "Project (this session):\n"
            f"- Name: {project.name}\n"
            f"- Root: {project.root}\n"
        )
        if project.description:
            proj += f"- Description: {project.description}\n"
        if project.project_context.strip():
            proj += "\n" + project.project_context.strip() + "\n"
        proj += (
            "\nAll file operations should target paths under the project root above "
            "unless explicitly asked to modify your workspace. The project is where "
            "the user can see and edit your work; your workspace is private scratch "
            "space. Uploads in this session land in `<project_root>/uploads/`.\n"
        )
        out += proj
    if contexts:
        joined = "\n\n".join(c.text for c in contexts if c.text.strip())
        if joined:
            out += (
                "\n\n---\n"
                "Session context (provided by the client for this session — "
                "additive, do not override the agent context above):\n"
                + joined
                + "\n"
            )
    if compaction_summary and compaction_summary.strip():
        out += (
            "\n\n---\n"
            "Prior conversation (summarized — this is your memory of everything "
            "that happened in this session before the messages that follow. Treat "
            "it as authoritative; the underlying turns have been dropped from "
            "your active context):\n"
            + compaction_summary.strip()
            + "\n"
        )
    return out


def append_context(conn: sqlite3.Connection, session_id: str, text: str) -> int:
    """Append a SessionContext message to a session. Returns the new total."""

    append_message(conn, session_id, SessionContext(text=text))
    return conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'session_context'",
        (session_id,),
    ).fetchone()[0]


def set_session_project(
    conn: sqlite3.Connection,
    session_id: str,
    new_project_id: str | None,
) -> tuple[Project | None, Project | None, int] | None:
    """Change a session's project binding. Returns
    ``(from_project, to_project, marker_row_id)`` on a real change, or
    ``None`` when the assignment is unchanged (idempotent no-op — caller
    can treat as success without emitting a marker).

    Also appends a `ProjectAssignmentChanged` marker to session history so
    the next turn's LLM message list shows the transition, and clients can
    render a "project changed" divider in the timeline. `marker_row_id` is
    the persisted row's `messages.id`; the endpoint attaches it as
    `event_id` on the broker `session_project_changed` frame.

    Callers should have already validated: session exists, agent owns it,
    the new project exists and is not soft-deleted, and no pending tool
    calls in history."""

    row = conn.execute(
        "SELECT project_id FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"session {session_id} not found")
    current_project_id: str | None = row["project_id"]
    if current_project_id == new_project_id:
        return None  # idempotent no-op

    from_project = (
        projects.get(conn, current_project_id) if current_project_id else None
    )
    to_project = projects.get(conn, new_project_id) if new_project_id else None

    conn.execute(
        "UPDATE sessions SET project_id = ? WHERE id = ?",
        (new_project_id, session_id),
    )
    marker_id = append_message(
        conn,
        session_id,
        ProjectAssignmentChanged(
            from_project_id=current_project_id,
            to_project_id=new_project_id,
            from_project_name=from_project.name if from_project else None,
            to_project_name=to_project.name if to_project else None,
            from_root=from_project.root if from_project else None,
            to_root=to_project.root if to_project else None,
            changed_at=now_ms(),
        ),
    )
    return from_project, to_project, marker_id


def classify_provider_error(exc: Exception) -> tuple[str, str]:
    """Map a provider exception to (code, message) for client surfacing.

    Codes: 'context_too_long' | 'rate_limit' | 'auth' | 'other'. The match is
    intentionally loose — uses both exception class name and message text so
    it works across Anthropic, OpenAI, OpenRouter (OpenAI-shaped), and Google
    without coupling to their import paths.

    The returned message always carries the exception class name plus any
    structured detail the SDK exposes (status code, request id). Without
    this, a bare provider 5xx surfaces as "Internal Server Error" with no
    breadcrumbs; with it, the same error becomes
    "APIStatusError [status=500]: Internal Server Error (request_id=req_...)".
    """

    name = type(exc).__name__
    raw = str(exc)
    low = raw.lower()
    message = _format_provider_error_message(exc, name, raw)

    context_hits = (
        "context_length_exceeded",
        "context length",
        "prompt is too long",
        "input is too long",
        "input token count",
        "exceeds the model's context",
        "exceeds the maximum context",
        "maximum context length",
    )
    if any(k in low for k in context_hits):
        return "context_too_long", message
    if "RateLimit" in name or "rate limit" in low or "rate_limit" in low or "429" in raw:
        return "rate_limit", message
    if (
        "Authentication" in name
        or "Unauthorized" in name
        or "401" in raw
        or "invalid api key" in low
        or "invalid_api_key" in low
    ):
        return "auth", message
    return "other", message


def _format_provider_error_message(exc: Exception, name: str, raw: str) -> str:
    """Enrich a provider exception with structured detail via duck typing.

    We don't want to import anthropic/openai/google-genai exception classes
    here — that would couple the runtime to specific SDK versions. Instead
    we look for common attributes on the exception object (`status_code`,
    `request_id`, `response.headers`) that those SDKs happen to expose,
    and include whatever's present.
    """
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    if not request_id:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                headers = getattr(response, "headers", None) or {}
                # OpenAI and Anthropic both use `x-request-id`; some
                # gateways use bare `request-id`.
                request_id = (
                    headers.get("x-request-id")
                    or headers.get("request-id")
                    or headers.get("x-anthropic-request-id")
                )
            except Exception:  # noqa: BLE001
                request_id = None

    prefix = name
    if status_code is not None:
        prefix += f" [status={status_code}]"
    out = f"{prefix}: {raw}" if raw else prefix
    if request_id:
        out += f" (request_id={request_id})"
    return out


def _log_error_traceback(prefix: str) -> None:
    """Log the current exception's traceback to stderr with a session-scoped
    prefix. Called from error-handling branches so an operator running
    `docker compose logs ark | grep <session-id>` can find the whole stack
    even when the wire message is thin."""
    import traceback

    print(prefix, file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


_SUMMARIZER_SYSTEM_PROMPT = """You are producing a summary of a prior conversation to preserve context that will be dropped from the model's active memory.

Include:
- Names, facts, and decisions from the conversation.
- Files referenced by path (created, modified, uploaded, shared).
- Code snippets discussed or written — paraphrase the logic; preserve file paths and function names.
- Commitments the assistant made to the user.
- Open questions and next steps.
- Any tool use that produced significant results.

Omit:
- Persona instructions or environment facts (those are provided separately).
- Small talk or greetings.
- Verbatim repetition of long tool outputs — summarize their essence.

Be complete over concise. The assistant will rely on this summary as its only memory of what happened before, so err toward including detail. Do not preface with "Here is a summary" — just write the summary."""


# LLM-invisible message kinds. Persisted in history for audit/replay/telemetry,
# but stripped before the message list is sent to the provider. Compaction
# summaries are excluded because their text is folded into the system prompt
# via the compaction slice logic instead.
_LLM_EXCLUDED = (SessionContext, TurnMetrics, RunError, CompactionSummary)


def _coerce_tz(name: str | None) -> str:
    """Return a validated IANA zone name, or 'UTC' if `name` is None,
    empty, or not parseable. Centralises the per-turn fallback semantics
    so callers (date-marker + env stanza) can't disagree."""
    if not name:
        return "UTC"
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo(name)
        return name
    except Exception:  # noqa: BLE001
        return "UTC"


def _now_local_date(tz_name: str) -> str:
    """Current date in the named zone (ISO YYYY-MM-DD). Falls back to UTC
    if the zone can't be resolved (shouldn't happen if the caller used
    `_coerce_tz` first, but defensive)."""
    from datetime import datetime, timezone as _tz
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name)).date().isoformat()
    except Exception:  # noqa: BLE001
        return datetime.now(_tz.utc).date().isoformat()


def _maybe_insert_date_marker(
    conn: sqlite3.Connection, session_id: str, client_tz: str | None = None,
) -> DateMarkerEvent | None:
    """If this is a conversational session and the previous UserText's
    date (as computed in `client_tz`) differs from today's date (same
    zone), insert a DateMarker row and return a DateMarkerEvent carrying
    its row_id. Otherwise return None.

    `client_tz` is the IANA zone name the current turn's client supplied
    (via the `timezone` field on the `user_message` frame). Both ends of
    the comparison use this zone — "from the client's current frame of
    reference, has the date changed?" Falls back to UTC when None or
    unparseable (see `_coerce_tz`).

    Scope limits for v1 — see docs/sessions.md § Date markers:
    - Conversational sessions only (cron/heartbeat have their own temporal
      framing baked into their session kind).
    - Calendar-date granularity only (no elapsed-hour trigger).
    - Compared against the previous `UserText` row, not just any message
      (so cron-fired `post_to_session` activity between user turns doesn't
      hide the "user came back" signal).
    """
    from datetime import datetime, timezone as _tz

    tz_name = _coerce_tz(client_tz)
    kind_row = conn.execute(
        "SELECT kind FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if kind_row is None or kind_row["kind"] != "conversational":
        return None
    prev = conn.execute(
        "SELECT created_at FROM messages "
        "WHERE session_id = ? AND role = 'user' "
        "ORDER BY seq DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if prev is None:
        return None  # first turn — nothing to compare against

    # Interpret BOTH ends in the current turn's zone.
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = _tz.utc
        tz_name = "UTC"
    prev_date = datetime.fromtimestamp(prev["created_at"] / 1000, tz=tz).date()
    now_date = datetime.now(tz).date()
    if prev_date == now_date:
        return None
    elapsed = (now_date - prev_date).days
    marker = DateMarker(
        from_date=prev_date.isoformat(),
        to_date=now_date.isoformat(),
        elapsed_days=elapsed,
        timezone=tz_name,
    )
    row_id = append_message(conn, session_id, marker)
    return DateMarkerEvent(
        from_date=marker.from_date,
        to_date=marker.to_date,
        elapsed_days=marker.elapsed_days,
        timezone=marker.timezone,
        row_id=row_id,
    )


def _date_marker_notification(msg: DateMarker) -> str:
    """Render a DateMarker as the synthetic user turn the model sees in the
    LLM message list. Explicit about elapsed time + the recalibration ask
    so the model doesn't silently continue to treat prior turns' sense of
    "today" or "yesterday" as still valid. Uses the marker's own timezone
    (set at insertion time from the client's supplied TZ)."""
    days = msg.elapsed_days
    plural = "day" if days == 1 else "days"
    return (
        "[system notification: Time has passed since the last turn in this "
        f"session. The current date is {msg.to_date} ({msg.timezone}). The "
        f"previous turn was on {msg.from_date} ({days} {plural} ago). Your "
        "prior sense of \"today\", \"yesterday\", or recent events may be "
        "stale — recalibrate accordingly.]"
    )


def _rewrite_for_llm(messages: list[Message]) -> list[Message]:
    """Apply per-kind rewrites needed before a message list goes to a provider.

    - `ProjectAssignmentChanged` → synthetic UserText notification of the
      project transition.
    - `DateMarker` → synthetic UserText notification that time has passed
      since the last user turn.

    The original marker rows stay in history for audit + client rendering;
    only the LLM-facing message list is rewritten."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, ProjectAssignmentChanged):
            out.append(UserText(text=_project_change_notification(m)))
        elif isinstance(m, DateMarker):
            out.append(UserText(text=_date_marker_notification(m)))
        else:
            out.append(m)
    return out


def _latest_compaction(history: list[Message]) -> tuple[int, CompactionSummary] | None:
    """Return (index, msg) of the latest CompactionSummary in history, or None."""
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], CompactionSummary):
            return i, history[i]
    return None


def _project_change_notification(msg: ProjectAssignmentChanged) -> str:
    """Render a ProjectAssignmentChanged marker as the notification the LLM
    sees in the message list. Explicit about historicity so the model doesn't
    treat prior file references as still-in-scope."""

    def _describe(name: str | None, root: str | None) -> str:
        if name is None and root is None:
            return "no project assignment"
        return f"'{name or '(unnamed)'}' at {root or '(unknown path)'}"

    return (
        "[system notification: This session's project assignment changed.\n"
        f"Previously: {_describe(msg.from_project_name, msg.from_root)}\n"
        f"Now: {_describe(msg.to_project_name, msg.to_root)}\n"
        "Continue helping the user. References to files under the previous "
        "project are historical context, not the current working area. "
        "Uploads and project-scoped operations now target the new location.]"
    )


def has_pending_tool_calls(history: list[Message]) -> bool:
    """True if any ToolCall in history has no matching ToolResult — i.e. the
    session is mid-tool-loop. Compacting across this boundary would leave the
    LLM with a ToolResult referencing an id it can no longer see. Used to
    guard the manual compaction endpoint (proactive/reactive triggers already
    happen at safe moments by construction)."""
    seen_results: set[str] = set()
    pending: set[str] = set()
    for m in history:
        if isinstance(m, ToolCall):
            pending.add(m.id)
        elif isinstance(m, ToolResult):
            seen_results.add(m.call_id)
    return bool(pending - seen_results)


def _should_compact_proactive(
    history: list[Message], threshold: float, context_window: int | None
) -> tuple[bool, int | None]:
    """Decide if we should proactively compact before the next turn.

    Returns (should_compact, last_input_tokens). Uses the last observed
    TurnMetrics as the fill gauge — a slight undercount since new messages
    have arrived since, which is fine (the threshold sits below the true
    ceiling anyway).

    Skips when:
    - context_window is unknown (no denominator)
    - no TurnMetrics observed yet (first turn)
    - latest TurnMetrics predates the latest CompactionSummary (stale — we
      compacted since observing, so we don't know the current fill)
    - fewer than 6 messages have accumulated since the last compaction
      (compacting a tiny history is silly)
    """
    if context_window is None or context_window <= 0:
        return False, None
    last_metrics_idx: int | None = None
    last_metrics: TurnMetrics | None = None
    for i in range(len(history) - 1, -1, -1):
        if isinstance(history[i], TurnMetrics):
            last_metrics_idx, last_metrics = i, history[i]
            break
    if last_metrics is None:
        return False, None
    latest = _latest_compaction(history)
    if latest is not None and last_metrics_idx is not None and last_metrics_idx < latest[0]:
        return False, last_metrics.input_tokens
    since_start = len(history) - (latest[0] + 1) if latest is not None else len(history)
    if since_start < 6:
        return False, last_metrics.input_tokens
    fraction = last_metrics.input_tokens / context_window
    if fraction < threshold:
        return False, last_metrics.input_tokens
    return True, last_metrics.input_tokens


async def compact_session(
    *,
    conn: sqlite3.Connection,
    config: Config,
    agent: AgentConfig,
    session_id: str,
    reason: str,
    exclude_last: int = 0,
    provider_factory=None,
) -> AsyncIterator[RuntimeEvent]:
    """Summarize prior conversation and persist a CompactionSummary row.

    Yields: CompactionStartedEvent, then either CompactionCompletedEvent
    or CompactionFailedEvent.

    `exclude_last` is the number of tail messages to hold out of the summary
    input — used by the reactive path to exclude the just-appended user
    message (which should show up in the retry, not in the summary).
    """
    context_window = models.context_window_for(agent.model, agent.max_context_tokens)
    yield CompactionStartedEvent(
        reason=reason, context_window=context_window, model=agent.model
    )

    history = load_history(conn, session_id)
    latest = _latest_compaction(history)
    prior_summary = latest[1].text if latest is not None else None
    slice_idx = latest[0] + 1 if latest is not None else 0

    # Post-slice content that will be summarized. Also strip LLM-invisible kinds
    # so the summarizer doesn't waste tokens on telemetry rows.
    to_summarize: list[Message] = _rewrite_for_llm(
        [m for m in history[slice_idx:] if not isinstance(m, _LLM_EXCLUDED)]
    )
    if exclude_last > 0:
        to_summarize = to_summarize[:-exclude_last]

    if not to_summarize:
        yield CompactionFailedEvent(
            code="other", message="nothing to summarize", reason=reason
        )
        return

    system = _SUMMARIZER_SYSTEM_PROMPT
    if prior_summary:
        system += (
            "\n\nPrior summary of context before the excerpt below "
            "(preserve information from it in your new summary):\n"
            + prior_summary
        )

    # Append a synthetic user turn asking for the summary. Without this,
    # message lists that end with an AssistantText leave the model waiting
    # for the "next" user turn — Gemini in particular returns empty text
    # rather than treating the system prompt's instruction as the ask.
    to_summarize = to_summarize + [
        UserText(
            text="Produce the summary now, as instructed in your system prompt."
        )
    ]

    provider_cfg = config.providers[agent.provider]
    # Resolve lazily so monkeypatching runtime.make_provider from tests works
    # (the default-arg pattern would capture the original function at
    # definition time).
    factory = provider_factory or make_provider
    provider = factory(
        provider_cfg.provider_type,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
    )

    summary_text = ""
    try:
        async for evt in provider.stream_turn(
            model=agent.model,
            system=system,
            messages=to_summarize,
            tools=[],
        ):
            if isinstance(evt, TextDelta):
                summary_text += evt.text
            # We intentionally do not forward the summarizer's own token
            # metrics or turn-end events — they'd be confusing telemetry
            # attributed to a "turn" that doesn't exist from the user's
            # perspective.
    except Exception as exc:  # noqa: BLE001
        code, message = classify_provider_error(exc)
        _log_error_traceback(
            f"[runtime] compaction failed for session {session_id} (code={code}, reason={reason}):"
        )
        yield CompactionFailedEvent(code=code, message=message, reason=reason)
        return

    summary_text = summary_text.strip()
    if not summary_text:
        yield CompactionFailedEvent(
            code="other", message="summarizer returned empty text", reason=reason
        )
        return

    summary_row_id = append_message(
        conn, session_id, CompactionSummary(text=summary_text, reason=reason)
    )
    yield CompactionCompletedEvent(
        summary=summary_text, reason=reason, row_id=summary_row_id
    )


# ---------------------------------------------------------------------------
# Turn loop
# ---------------------------------------------------------------------------


async def run_user_turn(
    *,
    conn: sqlite3.Connection,
    config: Config,
    agent: AgentConfig,
    session_id: str,
    user_text: str,
    provider_factory=None,
    max_tokens: int | None = None,
    client_timezone: str | None = None,
) -> AsyncIterator[RuntimeEvent]:
    """Persist the user message, then drive the model → tools → model loop.

    The loop terminates when either the model stops calling tools (natural end)
    or when the cumulative token budget (input + output summed across
    iterations of THIS turn — compaction is not counted) exceeds
    `max_tokens`. Budget precedence: explicit arg > agent.max_turn_tokens
    > DEFAULT_TURN_TOKEN_BUDGET.

    `client_timezone` is the IANA zone name the client supplied on this
    turn's `user_message` frame (or None/invalid → UTC). Used for the
    DateMarker comparison and the Environment stanza's "today's date"
    line. Falls back to UTC via `_coerce_tz`.
    """

    # Late-bound provider_factory default (mirrors compact_session) — resolves
    # the module attribute at call time so tests can inject via a
    # `runtime.make_provider` monkeypatch. The def-time capture would have
    # silently ignored that patch.
    if provider_factory is None:
        provider_factory = make_provider

    # Resolve the effective budget once at turn start.
    effective_budget = (
        max_tokens
        if max_tokens is not None
        else (agent.max_turn_tokens if agent.max_turn_tokens is not None else DEFAULT_TURN_TOKEN_BUDGET)
    )

    context_window = models.context_window_for(
        agent.model, agent.max_context_tokens
    )

    # Proactive compaction: check before persisting the user message so that
    # message ends up as the FIRST post-compaction turn — cleaner semantics
    # than "user message, intervening compaction, then a turn."
    compaction_used = False
    pre_history = load_history(conn, session_id)
    should_compact, last_input_tokens = _should_compact_proactive(
        pre_history, agent.compaction_threshold, context_window
    )
    if should_compact:
        reason = f"auto:threshold({last_input_tokens}/{context_window})"
        if agent.compaction_enabled:
            success = False
            async for evt in compact_session(
                conn=conn, config=config, agent=agent, session_id=session_id,
                reason=reason, provider_factory=provider_factory,
            ):
                yield evt
                if isinstance(evt, CompactionCompletedEvent):
                    success = True
            if success:
                compaction_used = True
        else:
            yield CompactionSkippedEvent(
                reason=f"disabled:{reason}",
                input_tokens=last_input_tokens,
                context_window=context_window,
            )

    # Late-bound default: resolve the module attribute at call time so tests
    # can inject a stub via `runtime.make_provider` (the def-time default
    # captured the original function and silently ignored that patch).
    if provider_factory is None:
        provider_factory = make_provider

    # Date-change marker: if this is a conversational session and the
    # calendar date (in the client's zone) differs from the previous
    # UserText's date, insert a DateMarker before the new user turn.
    # Model sees "[system notification: time has passed…]" via
    # `_rewrite_for_llm`; clients see a `DateMarker` row in history + a
    # `date_marker` wire event. See docs/sessions.md.
    date_marker_evt = _maybe_insert_date_marker(
        conn, session_id, client_tz=client_timezone
    )
    if date_marker_evt is not None:
        yield date_marker_evt

    append_message(conn, session_id, UserText(text=user_text))

    provider_cfg = config.providers[agent.provider]
    provider = provider_factory(
        provider_cfg.provider_type,
        api_key=provider_cfg.api_key,
        base_url=provider_cfg.base_url,
    )
    skills_for_session = loaded_skills(session_id)
    project = session_project(conn, session_id)

    last_stop_reason: str | None = None
    tokens_used = 0  # cumulative input+output across THIS turn's iterations
    while True:
        history = load_history(conn, session_id)
        # Slice for the LLM's message list at the latest CompactionSummary:
        # everything before it has been summarized and folded into the system
        # prompt below. SessionContext is timeless (persona-layer) and drawn
        # from the FULL history, not the slice.
        contexts = [m for m in history if isinstance(m, SessionContext)]
        latest = _latest_compaction(history)
        compaction_text: str | None = None
        if latest is not None:
            compaction_text = latest[1].text
            slice_history = history[latest[0] + 1:]
        else:
            slice_history = history
        turn_messages = _rewrite_for_llm(
            [m for m in slice_history if not isinstance(m, _LLM_EXCLUDED)]
        )
        system = system_prompt(
            agent, contexts, project=project, compaction_summary=compaction_text,
            client_timezone=client_timezone,
        )
        active = tools.active_schemas(agent, skills_for_session)
        pending_tool_calls: list[ToolCallEvent] = []
        turn_text = ""

        try:
            async for evt in provider.stream_turn(
                model=agent.model,
                system=system,
                messages=turn_messages,
                tools=active,
                # Per-response output cap the provider gets on THIS call
                # (SDK-level `max_tokens` kwarg). Distinct from `max_tokens`
                # on run_user_turn (whole-turn token budget). Falls back to
                # 4096 to match every provider adapter's def-time default.
                max_tokens=agent.max_output_tokens or 4096,
            ):
                if isinstance(evt, TextDelta):
                    turn_text += evt.text
                    yield evt
                elif isinstance(evt, ThinkingDelta):
                    yield evt
                elif isinstance(evt, ToolCallEvent):
                    # Live tool_call events go out without row_id: the row is
                    # persisted at TurnEnd (below), and we intentionally
                    # preserve today's DB ordering (AssistantText before
                    # ToolCall in `seq`) so provider adapters that
                    # reconstruct assistant blocks see the same shape they
                    # always have. Clients dedupe against the persisted-row
                    # events (assistant_message, tool_result) instead.
                    pending_tool_calls.append(evt)
                    yield evt
                elif isinstance(evt, TurnUsageEvent):
                    metrics_id = append_message(
                        conn,
                        session_id,
                        TurnMetrics(
                            input_tokens=evt.input_tokens,
                            output_tokens=evt.output_tokens,
                            model=evt.model or agent.model,
                        ),
                    )
                    tokens_used += evt.input_tokens + evt.output_tokens
                    # Re-emit with the agent's known context_window so the
                    # client can show a percentage. row_id lets clients
                    # advance their durable cursor from the live stream.
                    yield TurnUsageEvent(
                        input_tokens=evt.input_tokens,
                        output_tokens=evt.output_tokens,
                        model=evt.model or agent.model,
                        context_window=context_window,
                        row_id=metrics_id,
                    )
                elif isinstance(evt, AssistantTurnEnd):
                    last_stop_reason = evt.stop_reason
                    text_id: int | None = None
                    if turn_text:
                        text_id = append_message(
                            conn, session_id, AssistantText(text=turn_text)
                        )
                    for tc in pending_tool_calls:
                        append_message(
                            conn,
                            session_id,
                            ToolCall(
                                id=tc.id,
                                name=tc.name,
                                input=tc.input,
                                thought_signature=tc.thought_signature,
                            ),
                        )
                    yield AssistantTurnEnd(
                        text=turn_text,
                        stop_reason=evt.stop_reason,
                        row_id=text_id,
                    )
        except Exception as exc:  # noqa: BLE001
            code, message = classify_provider_error(exc)
            # Reactive compaction: if the provider rejects for context length
            # AND we haven't already compacted this turn AND the last message
            # is a fresh UserText (i.e. we're at turn start, not mid-tool-loop
            # — compacting across an unmatched ToolCall/ToolResult boundary
            # would confuse the retry), summarize and retry the same iteration.
            if (
                code == "context_too_long"
                and not compaction_used
                and agent.compaction_enabled
            ):
                last_msg_check = load_history(conn, session_id)
                if last_msg_check and isinstance(last_msg_check[-1], UserText):
                    # Rewind the user message so the CompactionSummary can land
                    # BEFORE it (and thus the retry's slice still sees the user
                    # message). We re-append after compaction — the message
                    # then becomes the first post-summary turn.
                    conn.execute(
                        "DELETE FROM messages WHERE session_id = ? AND seq = "
                        "(SELECT MAX(seq) FROM messages WHERE session_id = ?)",
                        (session_id, session_id),
                    )
                    compaction_used = True
                    success = False
                    async for c_evt in compact_session(
                        conn=conn, config=config, agent=agent, session_id=session_id,
                        reason="reactive:context_too_long",
                        provider_factory=provider_factory,
                    ):
                        yield c_evt
                        if isinstance(c_evt, CompactionCompletedEvent):
                            success = True
                    # Whether or not compaction succeeded, restore the user
                    # message — either the retry needs it, or the RunError we're
                    # about to persist needs the session to look consistent.
                    append_message(conn, session_id, UserText(text=user_text))
                    if success:
                        continue  # retry this iteration with compacted history
            _log_error_traceback(
                f"[runtime] turn error in session {session_id} (code={code}):"
            )
            err_id = append_message(
                conn, session_id, RunError(code=code, message=message)
            )
            yield RunErrorEvent(code=code, message=message, row_id=err_id)
            yield RunEnd(stop_reason=f"error:{code}")
            return

        if not pending_tool_calls:
            yield RunEnd(stop_reason=last_stop_reason)
            return

        ctx = tools.ToolContext(
            conn=conn,
            config=config,
            agent=agent,
            session_id=session_id,
            cwd=agent.workspace,
            loaded_skills=skills_for_session,
            metadata=session_metadata(conn, session_id),
        )
        for tc in pending_tool_calls:
            # Clear any stale attachments before dispatch; the tool may append
            # fresh entries via `current_context().pending_attachments` (see
            # the view_media built-in). We drain after execute returns.
            ctx.pending_attachments = []
            output, is_error = await tools.execute(tc.name, tc.input, ctx=ctx)
            attachments = list(ctx.pending_attachments or [])
            ctx.pending_attachments = []
            result_id = append_message(
                conn,
                session_id,
                ToolResult(
                    call_id=tc.id, output=output, is_error=is_error,
                    name=tc.name, attachments=attachments,
                ),
            )
            yield ToolResultEvent(
                call_id=tc.id, output=output, is_error=is_error, row_id=result_id
            )

        # Budget check between iterations. At least one iteration always runs
        # (the check happens AFTER the first iteration's TurnMetrics lands);
        # runaway loops are cut off at the boundary between iterations.
        if tokens_used >= effective_budget:
            message = (
                f"turn used {tokens_used} cumulative input+output tokens; "
                f"budget was {effective_budget}"
            )
            budget_err_id = append_message(
                conn, session_id,
                RunError(code="token_budget_exceeded", message=message),
            )
            yield RunErrorEvent(
                code="token_budget_exceeded",
                message=message,
                row_id=budget_err_id,
            )
            yield RunEnd(stop_reason="error:token_budget_exceeded")
            return


# ---------------------------------------------------------------------------
# In-flight turn registry (mine-capstone#485)
# ---------------------------------------------------------------------------

# session_id -> the asyncio.Task driving its in-flight turn. Registered by
# run_and_publish itself (asyncio.current_task on entry), so every spawn
# site — the WS handler, the scheduler — gets stop support without changes.
# Ark has no per-session concurrency guard; if a client violates the
# one-turn-at-a-time protocol the newest turn owns the slot.
_active_turns: dict[str, "asyncio.Task"] = {}


def stop_turn(session_id: str) -> bool:
    """Cancel the in-flight turn for a session (the `stop` WS command).

    Returns True when a running turn was cancelled. The cancelled task
    publishes a terminal ``done {"stopped": true}`` itself, so every
    subscriber sees the turn close.
    """

    task = _active_turns.get(session_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


# ---------------------------------------------------------------------------
# Wire-format conversion + broker publishing
# ---------------------------------------------------------------------------


def event_to_wire(evt: RuntimeEvent | Message) -> dict:
    """Convert a RuntimeEvent (or persisted Message) to its wire-format dict.

    Lives in runtime.py rather than server.py so both the WebSocket handler
    and the scheduler can use it without circular imports.

    Events corresponding to a persisted `messages` row carry `event_id`
    (matching `messages.id`) so downstream clients can maintain a durable
    cursor across the live WS and REST catch-up paths. Ephemeral events
    (TextDelta, ThinkingDelta, RunEnd, lifecycle-only compaction frames,
    the outer-catch error path) omit `event_id` — the client contract is
    "no event_id → don't advance cursor."
    """

    if isinstance(evt, TextDelta):
        return {"type": "assistant_delta", "text": evt.text}
    if isinstance(evt, ThinkingDelta):
        return {"type": "thinking", "delta": evt.text}
    if isinstance(evt, AssistantTurnEnd):
        out = {"type": "assistant_message", "text": evt.text}
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, ToolCallEvent):
        # Live tool_call frames are persisted at TurnEnd, so no event_id at
        # the point they go out. Clients dedupe on the tool-call correlation
        # id (`id`, matching the eventual tool_result frame's `id`), or fall
        # back to catch-up for the durable identity.
        return {"type": "tool_call", "id": evt.id, "name": evt.name, "input": evt.input}
    if isinstance(evt, ToolResultEvent):
        out = {
            "type": "tool_result",
            "id": evt.call_id,
            "output": evt.output,
            "error": evt.is_error,
        }
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, TurnUsageEvent):
        out = {
            "type": "turn_usage",
            "input_tokens": evt.input_tokens,
            "output_tokens": evt.output_tokens,
            "model": evt.model,
            "context_window": evt.context_window,
        }
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, RunErrorEvent):
        out = {"type": "error", "code": evt.code, "message": evt.message}
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, CompactionStartedEvent):
        return {
            "type": "compaction_started",
            "reason": evt.reason,
            "input_tokens": evt.input_tokens,
            "context_window": evt.context_window,
            "model": evt.model,
        }
    if isinstance(evt, CompactionCompletedEvent):
        out = {
            "type": "compaction_completed",
            "summary": evt.summary,
            "reason": evt.reason,
        }
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, CompactionFailedEvent):
        return {
            "type": "compaction_failed",
            "code": evt.code,
            "message": evt.message,
            "reason": evt.reason,
        }
    if isinstance(evt, CompactionSkippedEvent):
        return {
            "type": "compaction_skipped",
            "reason": evt.reason,
            "input_tokens": evt.input_tokens,
            "context_window": evt.context_window,
        }
    if isinstance(evt, DateMarkerEvent):
        out = {
            "type": "date_marker",
            "from_date": evt.from_date,
            "to_date": evt.to_date,
            "elapsed_days": evt.elapsed_days,
            "timezone": evt.timezone,
        }
        if evt.row_id is not None:
            out["event_id"] = evt.row_id
        return out
    if isinstance(evt, RunEnd):
        return {"type": "done", "stop_reason": evt.stop_reason}
    if is_dataclass(evt):
        return {"type": type(evt).__name__, **asdict(evt)}
    return {"type": "unknown"}


async def run_and_publish(
    *,
    conn: sqlite3.Connection,
    config: Config,
    agent: AgentConfig,
    session_id: str,
    user_text: str,
    max_tokens: int | None = None,
    client_timezone: str | None = None,
) -> None:
    """Drive a user turn and publish each event to the broker.

    Every event is tagged with `session_id` and `agent_name` so subscribers
    (per-session or global) can route. Use this from any code path that wants
    a turn's events visible to connected clients — the unified WS handler,
    the scheduler, etc.

    `max_tokens` overrides the per-turn token budget for this call. When None
    (the default), the resolution falls back to agent.max_turn_tokens, then
    DEFAULT_TURN_TOKEN_BUDGET.

    `client_timezone` is the IANA zone the client supplied on the
    `user_message` frame (or None → UTC fallback). See `run_user_turn`.
    """

    task = asyncio.current_task()
    if task is not None:
        _active_turns[session_id] = task
    try:
        async for evt in run_user_turn(
            conn=conn,
            config=config,
            agent=agent,
            session_id=session_id,
            user_text=user_text,
            max_tokens=max_tokens,
            client_timezone=client_timezone,
        ):
            wire = event_to_wire(evt)
            wire["session_id"] = session_id
            wire["agent_name"] = agent.name
            broker.publish(session_id, wire)
    except asyncio.CancelledError:
        # Real mid-turn cancel (`stop`, mine-capstone#485): close the turn
        # with a terminal event so clients aren't left mid-stream, then let
        # the cancellation land (the task ends cancelled).
        broker.publish(
            session_id,
            {
                "type": "done",
                "stop_reason": "stopped",
                "stopped": True,
                "session_id": session_id,
                "agent_name": agent.name,
            },
        )
        raise
    except Exception as e:  # noqa: BLE001
        # run_user_turn catches provider exceptions itself; this is for anything
        # that escapes (programming errors, broker failures, etc.). We persist
        # this as a RunError too — same shape as the in-turn error path — so
        # the error carries a durable event_id and shows up in /history and
        # catch-up. Closes the "no event_id → cursor stalls" gap for the
        # rare-but-real class of unhandled escapes.
        _log_error_traceback(
            f"[runtime] unhandled error in run_and_publish for session {session_id}:"
        )
        wire_msg = _format_provider_error_message(e, type(e).__name__, str(e))
        try:
            err_id = append_message(
                conn, session_id, RunError(code="other", message=wire_msg)
            )
        except Exception:  # noqa: BLE001
            # Best-effort: if even the DB write fails, publish without an id
            # rather than dropping the error frame entirely.
            err_id = None
        payload = {
            "type": "error",
            "session_id": session_id,
            "agent_name": agent.name,
            "code": "other",
            "message": wire_msg,
        }
        if err_id is not None:
            payload["event_id"] = err_id
        broker.publish(session_id, payload)
    finally:
        if task is not None and _active_turns.get(session_id) is task:
            del _active_turns[session_id]
