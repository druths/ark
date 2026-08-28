"""Session runtime: persistence, turn loop, tool dispatch."""

from __future__ import annotations

import json
import sqlite3
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
) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO sessions(id, agent_name, kind, created_at, project_id, cron_id) "
        "VALUES (?,?,?,?,?,?)",
        (sid, agent_name, kind, now_ms(), project_id, cron_id),
    )
    return sid


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


def append_message(conn: sqlite3.Connection, session_id: str, msg: Message) -> None:
    role, content = message_to_row(msg)
    next_seq = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?",
        (session_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO messages(session_id, seq, role, content_json, created_at) "
        "VALUES (?,?,?,?,?)",
        (session_id, next_seq, role, json.dumps(content), now_ms()),
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


def system_prompt(
    agent: AgentConfig,
    contexts: list[SessionContext] | None = None,
    project: Project | None = None,
    compaction_summary: str | None = None,
) -> str:
    """Build the system prompt.

    Layers (top to bottom):
      1. The user's `session_context.md` — agent identity / persona
      2. The Environment stanza — runtime facts (workspace path, available
         file/shell/upload helpers)
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
    env = (
        "\n\n---\n"
        "Environment (managed by the Ark harness, do not invent paths):\n"
        f"- Your name: {agent.name}\n"
        f"- Your workspace directory: {agent.workspace}\n"
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
        # Named blocks replace in place: the text of the LAST occurrence
        # renders at the position of the FIRST; unnamed blocks stay additive.
        latest_by_name: dict[str, str] = {}
        for c in contexts:
            if c.name:
                latest_by_name[c.name] = c.text
        effective: list[str] = []
        seen_names: set[str] = set()
        for c in contexts:
            if c.name:
                if c.name in seen_names:
                    continue
                seen_names.add(c.name)
                effective.append(latest_by_name[c.name])
            else:
                effective.append(c.text)
        joined = "\n\n".join(t for t in effective if t.strip())
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


def append_context(
    conn: sqlite3.Connection, session_id: str, text: str, name: str | None = None
) -> tuple[int, bool]:
    """Append a SessionContext message. Returns (total, replaced) — replaced
    is True when ``name`` matches an existing named block, which the system
    prompt then renders with THIS text at the original position (rows are
    append-only; the dedupe is prompt-build-time)."""

    replaced = False
    if name:
        history = load_history(conn, session_id)
        replaced = any(
            isinstance(m, SessionContext) and m.name == name for m in history
        )
    append_message(conn, session_id, SessionContext(text=text, name=name))
    total = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'session_context'",
        (session_id,),
    ).fetchone()[0]
    return total, replaced


def set_session_project(
    conn: sqlite3.Connection,
    session_id: str,
    new_project_id: str | None,
) -> tuple[Project | None, Project | None] | None:
    """Change a session's project binding. Returns (from_project, to_project)
    on a real change, or None when the assignment is unchanged (idempotent
    no-op — caller can treat as success without emitting a marker).

    Also appends a `ProjectAssignmentChanged` marker to session history so
    the next turn's LLM message list shows the transition, and clients can
    render a "project changed" divider in the timeline.

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
    append_message(
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
    return from_project, to_project


def classify_provider_error(exc: Exception) -> tuple[str, str]:
    """Map a provider exception to (code, message) for client surfacing.

    Codes: 'context_too_long' | 'rate_limit' | 'auth' | 'other'. The match is
    intentionally loose — uses both exception class name and message text so
    it works across Anthropic, OpenAI, OpenRouter (OpenAI-shaped), and Google
    without coupling to their import paths.
    """

    name = type(exc).__name__
    raw = str(exc)
    low = raw.lower()
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
        return "context_too_long", raw
    if "RateLimit" in name or "rate limit" in low or "rate_limit" in low or "429" in raw:
        return "rate_limit", raw
    if (
        "Authentication" in name
        or "Unauthorized" in name
        or "401" in raw
        or "invalid api key" in low
        or "invalid_api_key" in low
    ):
        return "auth", raw
    return "other", raw


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


def elide_tool_results(
    messages: list[Message],
    keep_turns: int = 2,
    elide_over: int = 2048,
    max_bytes: int = 65536,
) -> list[Message]:
    """In-memory tool-result elision (mine-capstone#697) — rows untouched.

    HYSTERESIS by design: nothing happens until the total in-view
    tool-result bytes exceed ``max_bytes``; then ONE pass replaces results
    older than the last ``keep_turns`` user turns that are larger than
    ``elide_over`` with a short stub naming the tool and call id. A
    per-turn trim would change the prompt prefix every turn and bust the
    provider prompt cache — one bulk pass keeps it warm between passes."""

    total = sum(
        len(m.output.encode("utf-8"))
        for m in messages
        if isinstance(m, ToolResult) and isinstance(m.output, str)
    )
    if total <= max_bytes:
        return messages
    # The index of the user turn ``keep_turns`` from the end — everything at
    # or after it is protected.
    user_indexes = [i for i, m in enumerate(messages) if isinstance(m, UserText)]
    protected_from = user_indexes[-keep_turns] if len(user_indexes) >= keep_turns else 0
    out: list[Message] = []
    for i, m in enumerate(messages):
        if (
            isinstance(m, ToolResult)
            and i < protected_from
            and isinstance(m.output, str)
            and len(m.output.encode("utf-8")) > elide_over
        ):
            size = len(m.output.encode("utf-8"))
            out.append(
                ToolResult(
                    call_id=m.call_id,
                    output=(
                        f"[tool result elided: {size} bytes; tool "
                        f"{m.name or 'unknown'}; call {m.call_id} — re-run "
                        "if needed]"
                    ),
                    is_error=m.is_error,
                    name=m.name,
                )
            )
        else:
            out.append(m)
    return out


def _rewrite_for_llm(messages: list[Message]) -> list[Message]:
    """Apply per-kind rewrites needed before a message list goes to a provider.

    Currently: substitute ProjectAssignmentChanged markers with synthetic
    UserText notifications so the model sees the transition as an event at
    that point in the timeline. (The original marker stays in history for
    audit + client rendering.)"""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, ProjectAssignmentChanged):
            out.append(UserText(text=_project_change_notification(m)))
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
            # The summarizer reuses the session's exact prefix — caching it
            # makes compaction nearly free on cache-capable providers.
            prompt_caching=agent.prompt_caching,
        ):
            if isinstance(evt, TextDelta):
                summary_text += evt.text
            # We intentionally do not forward the summarizer's own token
            # metrics or turn-end events — they'd be confusing telemetry
            # attributed to a "turn" that doesn't exist from the user's
            # perspective.
    except Exception as exc:  # noqa: BLE001
        code, message = classify_provider_error(exc)
        yield CompactionFailedEvent(code=code, message=message, reason=reason)
        return

    summary_text = summary_text.strip()
    if not summary_text:
        yield CompactionFailedEvent(
            code="other", message="summarizer returned empty text", reason=reason
        )
        return

    append_message(
        conn, session_id, CompactionSummary(text=summary_text, reason=reason)
    )
    yield CompactionCompletedEvent(summary=summary_text, reason=reason)


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
    provider_factory=make_provider,
    max_iterations: int = 16,
) -> AsyncIterator[RuntimeEvent]:
    """Persist the user message, then drive the model → tools → model loop."""

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
    for _ in range(max_iterations):
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
        turn_messages = elide_tool_results(
            turn_messages,
            keep_turns=agent.tool_result_keep_turns,
            elide_over=agent.tool_result_elide_over,
            max_bytes=agent.tool_result_max_bytes,
        )
        system = system_prompt(
            agent, contexts, project=project, compaction_summary=compaction_text
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
                prompt_caching=agent.prompt_caching,
            ):
                if isinstance(evt, TextDelta):
                    turn_text += evt.text
                    yield evt
                elif isinstance(evt, ThinkingDelta):
                    yield evt
                elif isinstance(evt, ToolCallEvent):
                    pending_tool_calls.append(evt)
                    yield evt
                elif isinstance(evt, TurnUsageEvent):
                    append_message(
                        conn,
                        session_id,
                        TurnMetrics(
                            input_tokens=evt.input_tokens,
                            output_tokens=evt.output_tokens,
                            model=evt.model or agent.model,
                            cached_input_tokens=evt.cached_input_tokens,
                            cache_write_tokens=evt.cache_write_tokens,
                        ),
                    )
                    # Re-emit with the agent's known context_window so the
                    # client can show a percentage.
                    yield TurnUsageEvent(
                        input_tokens=evt.input_tokens,
                        output_tokens=evt.output_tokens,
                        model=evt.model or agent.model,
                        context_window=context_window,
                        cached_input_tokens=evt.cached_input_tokens,
                        cache_write_tokens=evt.cache_write_tokens,
                    )
                elif isinstance(evt, AssistantTurnEnd):
                    last_stop_reason = evt.stop_reason
                    if turn_text:
                        append_message(conn, session_id, AssistantText(text=turn_text))
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
                    yield AssistantTurnEnd(text=turn_text, stop_reason=evt.stop_reason)
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
            append_message(conn, session_id, RunError(code=code, message=message))
            yield RunErrorEvent(code=code, message=message)
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
        )
        for tc in pending_tool_calls:
            output, is_error = await tools.execute(tc.name, tc.input, ctx=ctx)
            append_message(
                conn,
                session_id,
                ToolResult(
                    call_id=tc.id, output=output, is_error=is_error, name=tc.name
                ),
            )
            yield ToolResultEvent(call_id=tc.id, output=output, is_error=is_error)

    yield RunEnd(stop_reason="max_iterations")


# ---------------------------------------------------------------------------
# Wire-format conversion + broker publishing
# ---------------------------------------------------------------------------


def event_to_wire(evt: RuntimeEvent | Message) -> dict:
    """Convert a RuntimeEvent (or persisted Message) to its wire-format dict.

    Lives in runtime.py rather than server.py so both the WebSocket handler
    and the scheduler can use it without circular imports.
    """

    if isinstance(evt, TextDelta):
        return {"type": "assistant_delta", "text": evt.text}
    if isinstance(evt, ThinkingDelta):
        return {"type": "thinking", "delta": evt.text}
    if isinstance(evt, AssistantTurnEnd):
        return {"type": "assistant_message", "text": evt.text}
    if isinstance(evt, ToolCallEvent):
        return {"type": "tool_call", "id": evt.id, "name": evt.name, "input": evt.input}
    if isinstance(evt, ToolResultEvent):
        return {
            "type": "tool_result",
            "id": evt.call_id,
            "output": evt.output,
            "error": evt.is_error,
        }
    if isinstance(evt, TurnUsageEvent):
        return {
            "type": "turn_usage",
            "input_tokens": evt.input_tokens,
            "output_tokens": evt.output_tokens,
            "model": evt.model,
            "context_window": evt.context_window,
            "cached_input_tokens": evt.cached_input_tokens,
            "cache_write_tokens": evt.cache_write_tokens,
        }
    if isinstance(evt, RunErrorEvent):
        return {"type": "error", "code": evt.code, "message": evt.message}
    if isinstance(evt, CompactionStartedEvent):
        return {
            "type": "compaction_started",
            "reason": evt.reason,
            "input_tokens": evt.input_tokens,
            "context_window": evt.context_window,
            "model": evt.model,
        }
    if isinstance(evt, CompactionCompletedEvent):
        return {
            "type": "compaction_completed",
            "summary": evt.summary,
            "reason": evt.reason,
        }
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
) -> None:
    """Drive a user turn and publish each event to the broker.

    Every event is tagged with `session_id` and `agent_name` so subscribers
    (per-session or global) can route. Use this from any code path that wants
    a turn's events visible to connected clients — the unified WS handler,
    the scheduler, etc.
    """

    try:
        async for evt in run_user_turn(
            conn=conn,
            config=config,
            agent=agent,
            session_id=session_id,
            user_text=user_text,
        ):
            wire = event_to_wire(evt)
            wire["session_id"] = session_id
            wire["agent_name"] = agent.name
            broker.publish(session_id, wire)
    except Exception as e:  # noqa: BLE001
        # run_user_turn catches provider exceptions itself; this is for anything
        # that escapes (programming errors, broker failures, etc.).
        broker.publish(
            session_id,
            {
                "type": "error",
                "session_id": session_id,
                "agent_name": agent.name,
                "code": "other",
                "message": f"{type(e).__name__}: {e}",
            },
        )
