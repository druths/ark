"""Normalized message, tool, and stream event types.

These are the provider-agnostic shapes the rest of Ark works with. Each
provider adapter is responsible for translating to/from its native format.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Union

# ---------------------------------------------------------------------------
# Conversation messages (one row in the `messages` table = one of these)
# ---------------------------------------------------------------------------


@dataclass
class UserText:
    text: str


@dataclass
class AssistantText:
    text: str
    injected_from: str | None = None  # set when injected from another session via post_to_session


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]
    # Provider-specific opaque bytes that must be echoed back when this call
    # reappears in conversation history. Currently only Gemini 2.5+ thinking
    # models populate this — other providers leave it None.
    thought_signature: bytes | None = None


@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False
    name: str = ""  # name of the tool that produced this result (required by Google's API)
    # Optional media attachments the tool produced (image/pdf/audio/video).
    # Each entry: {"type": "image" | "pdf" | ..., "path": "<fs path>", "mime": "image/png"}.
    # Provider adapters read these at message-list build time and translate
    # to the provider's native tool_result-with-media content shape. See
    # the view_media built-in tool and docs/sessions.md § Multi-modal.
    attachments: list[dict] = field(default_factory=list)


@dataclass
class UploadMessage:
    """A file the client uploaded into the agent's workspace.

    Stored as user-side context in conversation history. The actual bytes
    live on disk at <workspace>/<path>; this message is just the record."""

    path: str  # workspace-relative
    original_name: str  # filename before any auto-suffix
    size: int


@dataclass
class SharedFile:
    """A file the agent has shared with the client via `share_with_client`.

    Stored as assistant-side context in conversation history. The bytes live
    on disk at <workspace>/<path>; clients fetch via the download endpoint."""

    path: str  # workspace-relative
    description: str = ""
    size: int = 0


@dataclass
class Project:
    """A shared, user-visible working directory that one or more sessions can be
    bound to.

    Unlike an agent's workspace (which is per-agent and private), a project's
    root is intended to be inspected and edited by clients (file browser /
    upload / edit) and watched for changes that fan out to live subscribers.
    Soft-deletable — `deleted_at` is set on delete, files on disk are not
    touched.
    """

    id: str
    name: str
    root: str            # absolute filesystem path
    description: str = ""
    project_context: str = ""  # appended to system prompt for project sessions
    created_at: int = 0
    deleted_at: int | None = None


@dataclass
class SessionContext:
    """Client-supplied per-session instructions, layered onto the system prompt.

    Append-only: multiple SessionContext messages accumulate over the life of
    the session. They are NOT sent to the LLM as conversation turns — the
    runtime extracts them and appends to the system prompt instead."""

    text: str


@dataclass
class TurnMetrics:
    """Per-turn telemetry: token counts reported by the provider.

    Persisted in session history so total session cost / context fill can be
    computed later. NOT sent to the LLM as a conversation turn — the runtime
    filters these out before passing the message list to the provider."""

    input_tokens: int
    output_tokens: int
    model: str = ""


@dataclass
class RunError:
    """A classified failure during a turn. Persisted so clients (and humans
    looking at the session later) can see what went wrong and where."""

    code: str  # one of: context_too_long, rate_limit, auth, other
    message: str


@dataclass
class DateMarker:
    """Persisted at turn-start when the calendar date differs from the
    previous UserText in the session — the "wake up, time has passed" cue
    for the model after a session has been idle for a day or more.

    `from_date`/`to_date` are ISO YYYY-MM-DD in `timezone` — the zone the
    current turn's client supplied (or "UTC" if none). Clients can render
    "── Oct 5 (LA time) ──" style dividers using this field.

    The row lives in history with its own kind so clients can render it as
    a timeline divider (or filter it out of the chat view, same policy as
    other system markers). The runtime substitutes it with a synthetic
    UserText notification via `_rewrite_for_llm` when building the LLM's
    message list, so the model sees an explicit "time has passed"
    notification at that point in the conversation.

    Only inserted on `conversational` sessions — cron/heartbeat have their
    own temporal framing."""

    from_date: str   # ISO YYYY-MM-DD of the previous UserText's date in `timezone`
    to_date: str     # ISO YYYY-MM-DD of the current date in `timezone`
    elapsed_days: int
    timezone: str = "UTC"  # IANA zone name used for the comparison


@dataclass
class ProjectAssignmentChanged:
    """Marker persisted at the moment a session's project assignment changes.

    Both endpoints are nullable: `to_project_id=None` means the session was
    detached from a project; `from_project_id=None` means it had no project
    before. Persisted for audit + client-side rendering; substituted for a
    synthetic UserText when the runtime builds the LLM's message list so the
    model sees the transition as an event at that point in the timeline.

    Once written, this row is immutable — subsequent reassignments produce
    new rows, giving an ordered assignment-history."""

    from_project_id: str | None
    to_project_id: str | None
    from_project_name: str | None = None
    to_project_name: str | None = None
    from_root: str | None = None
    to_root: str | None = None
    changed_at: int = 0


@dataclass
class CompactionSummary:
    """A summary of prior conversation, folded into the system prompt from
    this row's position forward.

    Persisted in history like any other message so clients can render the
    compaction as a visual divider and expose the summary text to the user.
    NOT sent to the LLM as a conversation turn — the runtime slices history
    at the latest CompactionSummary and folds its `text` into the system
    prompt as a new stanza.

    Multiple compactions accumulate. The runtime always uses the LATEST one
    as the slice point; older summaries stay in history as an audit trail
    of what got dropped and when."""

    text: str
    reason: str = ""  # e.g. "auto:threshold(0.87)", "reactive:context_too_long"


Message = Union[
    UserText,
    AssistantText,
    ToolCall,
    ToolResult,
    UploadMessage,
    SharedFile,
    SessionContext,
    TurnMetrics,
    RunError,
    CompactionSummary,
    ProjectAssignmentChanged,
    DateMarker,
]


# ---------------------------------------------------------------------------
# Tool schemas (what the LLM sees in its tool list)
# ---------------------------------------------------------------------------


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


# ---------------------------------------------------------------------------
# Stream events
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ToolCallEvent:
    id: str                # tool-call correlation id (matches ToolResultEvent.call_id)
    name: str
    input: dict[str, Any]
    thought_signature: bytes | None = None  # see ToolCall.thought_signature
    # messages.id of the corresponding ToolCall row (populated by the
    # runtime after append_message). See event_to_wire — surfaces as
    # `event_id` on the wire for durable-cursor dedupe.
    row_id: int | None = None


@dataclass
class ToolResultEvent:
    call_id: str
    output: str
    is_error: bool = False
    row_id: int | None = None  # messages.id of ToolResult row


@dataclass
class AssistantTurnEnd:
    """End of one provider turn (one stream_turn call)."""

    text: str  # full assembled assistant text for this turn
    stop_reason: str | None = None
    # messages.id of the corresponding AssistantText row. None when the
    # turn ended with no text (only tool calls, or an error mid-generation).
    row_id: int | None = None


@dataclass
class RunEnd:
    """End of the whole run loop (no more tool calls — the agent is done)."""

    stop_reason: str | None = None


@dataclass
class TurnUsageEvent:
    """Token counts reported by the provider for the just-completed turn.

    Yielded by provider adapters just before AssistantTurnEnd. The runtime
    persists this as a TurnMetrics message and forwards it to the client."""

    input_tokens: int
    output_tokens: int
    model: str = ""
    context_window: int | None = None  # provider's known max, if any
    row_id: int | None = None           # messages.id of TurnMetrics row


@dataclass
class RunErrorEvent:
    """Classified provider failure, surfaced to clients with an actionable code."""

    code: str  # one of: context_too_long, rate_limit, auth, other
    message: str
    row_id: int | None = None  # messages.id of RunError row


@dataclass
class CompactionStartedEvent:
    """Compaction is about to run. Clients can render "Compacting session…"
    UX while awaiting completed/failed."""

    reason: str            # e.g. "auto:threshold(0.87)", "reactive:context_too_long"
    input_tokens: int | None = None
    context_window: int | None = None
    model: str = ""


@dataclass
class CompactionCompletedEvent:
    """Compaction succeeded; CompactionSummary row persisted. Subsequent turns
    will see only the summary + post-compaction messages."""

    summary: str
    reason: str = ""
    row_id: int | None = None  # messages.id of CompactionSummary row


@dataclass
class CompactionFailedEvent:
    """Compaction attempt errored (summarizer call raised). The impending turn
    proceeds uncompacted and will likely fail with context_too_long, which is
    the existing recovery path."""

    code: str  # classified provider error code
    message: str
    reason: str = ""


@dataclass
class CompactionSkippedEvent:
    """Proactive threshold was crossed but compaction is disabled for this
    agent. Emitted once per turn to alert the client without acting."""

    reason: str            # "disabled:threshold(0.87)"
    input_tokens: int | None = None
    context_window: int | None = None


@dataclass
class DateMarkerEvent:
    """A DateMarker was inserted at the start of this turn — the calendar
    date changed (in `timezone`) since the previous user turn. Published
    to the broker so live WS clients can advance their durable cursor
    (via `event_id`) and render a date divider in their timeline."""

    from_date: str
    to_date: str
    elapsed_days: int
    timezone: str = "UTC"
    row_id: int | None = None  # messages.id of the DateMarker row


ProviderEvent = Union[
    TextDelta, ThinkingDelta, ToolCallEvent, AssistantTurnEnd, TurnUsageEvent
]
RuntimeEvent = Union[
    TextDelta,
    ThinkingDelta,
    ToolCallEvent,
    ToolResultEvent,
    AssistantTurnEnd,
    RunEnd,
    TurnUsageEvent,
    RunErrorEvent,
    CompactionStartedEvent,
    CompactionCompletedEvent,
    CompactionFailedEvent,
    CompactionSkippedEvent,
    DateMarkerEvent,
]


# ---------------------------------------------------------------------------
# JSON serialization for the messages table content_json column
# ---------------------------------------------------------------------------


def message_to_row(msg: Message) -> tuple[str, dict[str, Any]]:
    """Return (role, content_dict) for storage."""
    if isinstance(msg, UserText):
        return "user", {"text": msg.text}
    if isinstance(msg, AssistantText):
        body: dict[str, Any] = {"text": msg.text}
        if msg.injected_from:
            body["injected_from"] = msg.injected_from
        return "assistant", body
    if isinstance(msg, ToolCall):
        body: dict[str, Any] = {"id": msg.id, "name": msg.name, "input": msg.input}
        if msg.thought_signature:
            body["thought_signature_b64"] = base64.b64encode(msg.thought_signature).decode("ascii")
        return "tool_call", body
    if isinstance(msg, ToolResult):
        body: dict[str, Any] = {
            "call_id": msg.call_id,
            "output": msg.output,
            "is_error": msg.is_error,
        }
        if msg.name:
            body["name"] = msg.name
        if msg.attachments:
            body["attachments"] = list(msg.attachments)
        return "tool_result", body
    if isinstance(msg, UploadMessage):
        return "upload", {
            "path": msg.path,
            "original_name": msg.original_name,
            "size": msg.size,
        }
    if isinstance(msg, SharedFile):
        return "shared_file", {
            "path": msg.path,
            "description": msg.description,
            "size": msg.size,
        }
    if isinstance(msg, SessionContext):
        return "session_context", {"text": msg.text}
    if isinstance(msg, TurnMetrics):
        return "turn_metrics", {
            "input_tokens": msg.input_tokens,
            "output_tokens": msg.output_tokens,
            "model": msg.model,
        }
    if isinstance(msg, RunError):
        return "run_error", {"code": msg.code, "message": msg.message}
    if isinstance(msg, CompactionSummary):
        return "compaction_summary", {"text": msg.text, "reason": msg.reason}
    if isinstance(msg, ProjectAssignmentChanged):
        return "project_assignment_changed", {
            "from_project_id": msg.from_project_id,
            "to_project_id": msg.to_project_id,
            "from_project_name": msg.from_project_name,
            "to_project_name": msg.to_project_name,
            "from_root": msg.from_root,
            "to_root": msg.to_root,
            "changed_at": msg.changed_at,
        }
    if isinstance(msg, DateMarker):
        return "date_marker", {
            "from_date": msg.from_date,
            "to_date": msg.to_date,
            "elapsed_days": msg.elapsed_days,
            "timezone": msg.timezone,
        }
    raise TypeError(f"unknown message type: {type(msg).__name__}")


def message_from_row(role: str, content: dict[str, Any]) -> Message:
    if role == "user":
        return UserText(text=content["text"])
    if role == "assistant":
        return AssistantText(text=content["text"], injected_from=content.get("injected_from"))
    if role == "tool_call":
        sig_b64 = content.get("thought_signature_b64")
        sig = base64.b64decode(sig_b64) if sig_b64 else None
        return ToolCall(
            id=content["id"],
            name=content["name"],
            input=content["input"],
            thought_signature=sig,
        )
    if role == "tool_result":
        return ToolResult(
            call_id=content["call_id"],
            output=content["output"],
            is_error=content.get("is_error", False),
            name=content.get("name", ""),
            attachments=list(content.get("attachments") or []),
        )
    if role == "upload":
        return UploadMessage(
            path=content["path"],
            original_name=content["original_name"],
            size=content["size"],
        )
    if role == "shared_file":
        return SharedFile(
            path=content["path"],
            description=content.get("description", ""),
            size=content.get("size", 0),
        )
    if role == "session_context":
        return SessionContext(text=content["text"])
    if role == "turn_metrics":
        return TurnMetrics(
            input_tokens=int(content.get("input_tokens", 0)),
            output_tokens=int(content.get("output_tokens", 0)),
            model=content.get("model", ""),
        )
    if role == "run_error":
        return RunError(code=content.get("code", "other"), message=content.get("message", ""))
    if role == "compaction_summary":
        return CompactionSummary(
            text=content.get("text", ""), reason=content.get("reason", "")
        )
    if role == "project_assignment_changed":
        return ProjectAssignmentChanged(
            from_project_id=content.get("from_project_id"),
            to_project_id=content.get("to_project_id"),
            from_project_name=content.get("from_project_name"),
            to_project_name=content.get("to_project_name"),
            from_root=content.get("from_root"),
            to_root=content.get("to_root"),
            changed_at=int(content.get("changed_at", 0)),
        )
    if role == "date_marker":
        return DateMarker(
            from_date=content.get("from_date", ""),
            to_date=content.get("to_date", ""),
            elapsed_days=int(content.get("elapsed_days", 0)),
            timezone=content.get("timezone", "UTC"),
        )
    raise ValueError(f"unknown role: {role}")
