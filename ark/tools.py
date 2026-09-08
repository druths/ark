"""Built-in tools available to every agent, plus the tool-execution machinery.

Tools are kwargs-style callables. Anything they need from the runtime (the agent
they're running for, the session id, the DB connection, the live broker) comes
via `current_context()` — a contextvar set by the runtime just before dispatch.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import signal
import sqlite3
import subprocess
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .types import ToolSchema

if TYPE_CHECKING:
    from .config import AgentConfig, Config


# ---------------------------------------------------------------------------
# Tool context (set by the runtime before dispatch; read by tools that need it)
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    conn: sqlite3.Connection
    config: "Config"
    agent: "AgentConfig"
    session_id: str
    cwd: Path
    loaded_skills: set[str]  # mutable set of skill names loaded for this session
    # Client-supplied session metadata (create_session `metadata`): the same
    # unforgeable server-side channel as session_id — never model-visible.
    # Skills read per-session capabilities from here (e.g. a callback pair).
    metadata: dict | None = None


_context: ContextVar[ToolContext] = ContextVar("ark_tool_context")


def current_context() -> ToolContext:
    return _context.get()


class ToolError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


@dataclass
class BuiltinTool:
    schema: ToolSchema
    fn: Callable[..., Any]
    is_async: bool = False


BUILTINS: dict[str, BuiltinTool] = {}


def _register(schema: ToolSchema, fn: Callable[..., Any], is_async: bool = False) -> None:
    BUILTINS[schema.name] = BuiltinTool(schema=schema, fn=fn, is_async=is_async)


# ---------------------------------------------------------------------------
# File / shell tools
# ---------------------------------------------------------------------------


def _read_file(*, path: str) -> str:
    p = Path(path).expanduser()
    try:
        return p.read_text()
    except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
        raise ToolError(str(e))


def _write_file(*, path: str, content: str) -> str:
    p = Path(path).expanduser().resolve()
    # An agent must not be able to scribble into *other* agents' workspaces.
    # (run_command is intentionally exempt — it's the shell escape hatch.)
    ctx = current_context()
    for other_name, other_agent in ctx.config.agents.items():
        if other_name == ctx.agent.name:
            continue
        other_ws = Path(other_agent.workspace).resolve()
        try:
            p.relative_to(other_ws)
        except ValueError:
            continue
        raise ToolError(
            f"refusing to write to another agent's workspace "
            f"({other_name!r} owns {other_ws})"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"


def _list_files(*, path: str = ".", pattern: str | None = None) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"path not found: {p}")
    if p.is_file():
        return str(p)
    entries = sorted(p.iterdir())
    if pattern:
        entries = [e for e in entries if fnmatch.fnmatch(e.name, pattern)]
    if not entries:
        return "(empty)"
    return "\n".join(f"{e.name}{'/' if e.is_dir() else ''}" for e in entries)


# session_id -> in-flight run_command processes, so `stop` can terminate
# them (mine-capstone#485). Task cancellation alone cannot: the command runs
# in a to_thread worker, and cancelling the awaiting task leaves the thread
# — and the subprocess — running to its timeout.
_active_procs: dict[str, set[subprocess.Popen]] = {}
_KILL_GRACE_S = 5.0


def _kill_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def _escalate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        _kill_group(proc, signal.SIGKILL)


def stop_session_commands(session_id: str) -> int:
    """SIGTERM the process group of every in-flight run_command for the
    session, escalating to SIGKILL after a grace period (the design note's
    kill discipline). Returns the number of processes signalled. Safe from
    the event loop: signalling is instant, escalation rides a timer thread.
    """

    procs = [p for p in _active_procs.get(session_id, set()) if p.poll() is None]
    for proc in procs:
        _kill_group(proc, signal.SIGTERM)
        timer = threading.Timer(_KILL_GRACE_S, _escalate, args=(proc,))
        timer.daemon = True
        timer.start()
    return len(procs)


def _run_command(*, command: str, timeout_seconds: float = 60) -> str:
    timeout = min(float(timeout_seconds), 600)
    ctx = _context.get(None)
    session_id = ctx.session_id if ctx is not None else None
    # start_new_session: own process group, so stop/timeout kills the whole
    # tree (shell + children), not just the shell.
    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    if session_id is not None:
        _active_procs.setdefault(session_id, set()).add(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.communicate(timeout=_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGKILL)
            proc.communicate()
        raise ToolError(f"command timed out after {timeout}s") from e
    finally:
        if session_id is not None:
            _active_procs.get(session_id, set()).discard(proc)
    parts = []
    if stdout:
        parts.append(f"--- stdout ---\n{stdout.rstrip()}")
    if stderr:
        parts.append(f"--- stderr ---\n{stderr.rstrip()}")
    parts.append(f"exit code: {proc.returncode}")
    return "\n".join(parts)


_register(
    ToolSchema(
        name="read_file",
        description="Read the full contents of a file from disk.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    _read_file,
)

_register(
    ToolSchema(
        name="write_file",
        description="Write content to a file (overwrites existing). Creates parent dirs.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    ),
    _write_file,
)

_register(
    ToolSchema(
        name="list_files",
        description="List files and directories at the given path. Optional glob pattern filters by name.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "pattern": {"type": "string"},
            },
        },
    ),
    _list_files,
)

_register(
    ToolSchema(
        name="run_command",
        description="Run a shell command and return stdout, stderr, and exit code.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "number", "default": 60},
            },
            "required": ["command"],
        },
    ),
    _run_command,
)


# ---------------------------------------------------------------------------
# Internet: search + URL fetch
# ---------------------------------------------------------------------------


async def _search_web(*, query: str, count: int = 10) -> str:
    ctx = current_context()
    tools_cfg = ctx.config.tools or {}
    search_cfg = tools_cfg.get("search_web") or {}
    provider = (search_cfg.get("provider") or "brave").lower()
    if provider == "brave":
        return await _search_web_brave(tools_cfg, query, count)
    if provider == "tavily":
        return await _search_web_tavily(tools_cfg, query, count)
    raise ToolError(
        f"unknown search_web provider {provider!r}. Configure tools.search_web.provider "
        "to one of: 'brave', 'tavily'."
    )


async def _search_web_brave(tools_cfg: dict, query: str, count: int) -> str:
    import re as _re

    import httpx as _httpx

    cfg = tools_cfg.get("brave_search") or {}
    api_key = cfg.get("api_key")
    if not api_key:
        raise ToolError(
            "search_web (brave) requires `tools.brave_search.api_key` to be set in config.json"
        )
    count = max(1, min(int(count), 20))
    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                },
                params={"q": query, "count": count},
            )
    except (_httpx.NetworkError, _httpx.TimeoutException, _httpx.HTTPError) as e:
        raise ToolError(f"brave_search request failed: {type(e).__name__}: {e}")
    if r.status_code >= 400:
        raise ToolError(f"brave_search HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    results = ((data.get("web") or {}).get("results") or [])[:count]
    if not results:
        return "(no results)"
    lines = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        desc = _re.sub(r"<[^>]+>", "", item.get("description") or "").strip()
        lines.append(f"- {title}\n  {url}\n  {desc}")
    return "\n".join(lines)


async def _search_web_tavily(tools_cfg: dict, query: str, count: int) -> str:
    import httpx as _httpx

    vendor = tools_cfg.get("tavily") or {}
    api_key = vendor.get("api_key")
    if not api_key:
        raise ToolError(
            "search_web (tavily) requires `tools.tavily.api_key` to be set in config.json"
        )
    search_depth = vendor.get("search_depth", "basic")
    count = max(1, min(int(count), 20))
    body = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "max_results": count,
    }
    try:
        async with _httpx.AsyncClient(timeout=20) as client:
            r = await client.post("https://api.tavily.com/search", json=body)
    except (_httpx.NetworkError, _httpx.TimeoutException, _httpx.HTTPError) as e:
        raise ToolError(f"tavily_search request failed: {type(e).__name__}: {e}")
    if r.status_code >= 400:
        raise ToolError(f"tavily_search HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    results = (data.get("results") or [])[:count]
    if not results:
        return "(no results)"
    lines = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        # Tavily's `content` is already cleaned text; truncate for digestibility.
        snippet = (item.get("content") or "").strip()
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(f"- {title}\n  {url}\n  {snippet}")
    return "\n".join(lines)


async def _fetch_url(*, url: str, timeout_seconds: float = 15) -> str:
    from . import resolvers

    ctx = current_context()
    tools_cfg = ctx.config.tools or {}
    cfg = tools_cfg.get("fetch_url") or {}
    spec = cfg.get("resolver_sequence")
    try:
        # `tools_cfg` doubles as vendor blocks: e.g. `tools.tavily.api_key` is
        # picked up by a `{"provider": "tavily"}` resolver entry that doesn't
        # specify its own api_key. Per-entry config still wins on conflict.
        chain = resolvers.build_chain(spec, vendor_blocks=tools_cfg)
    except resolvers.ResolverConfigError as e:
        raise ToolError(f"fetch_url misconfigured: {e}")
    # Per-resolver timeout; whole-chain budget is bounded since each resolver
    # uses its own AsyncClient timeout, so the worst-case wall clock is
    # roughly len(chain) * timeout.
    result = await resolvers.fetch_with_chain(
        url, chain, timeout=float(timeout_seconds)
    )
    if not result.ok:
        raise ToolError(result.content)
    return result.content


_register(
    ToolSchema(
        name="search_web",
        description="Run an internet search via the Brave Search API. Returns the top results as title/URL/snippet triples — follow up on anything that looks relevant with `fetch_url`.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of results to return (1–20).",
                },
            },
            "required": ["query"],
        },
    ),
    _search_web,
    is_async=True,
)

_register(
    ToolSchema(
        name="fetch_url",
        description="Fetch a URL and return its readable content. Tries a configured chain of resolvers (e.g. plain httpx first, then Jina Reader for JS-heavy pages). Returns markdown or extracted text; truncates at 1 MB.",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "timeout_seconds": {"type": "number", "default": 15},
            },
            "required": ["url"],
        },
    ),
    _fetch_url,
    is_async=True,
)


# ---------------------------------------------------------------------------
# Schedule meta-tools — manage the current agent's heartbeat and crons
# ---------------------------------------------------------------------------


def _set_heartbeat(*, seconds: float | None) -> str:
    ctx = current_context()
    if seconds is None or float(seconds) <= 0:
        ctx.conn.execute(
            "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, NULL) "
            "ON CONFLICT(agent_name) DO UPDATE SET heartbeat_seconds = NULL",
            (ctx.agent.name,),
        )
        return f"heartbeat disabled for {ctx.agent.name}"
    s = int(seconds)
    ctx.conn.execute(
        "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, ?) "
        "ON CONFLICT(agent_name) DO UPDATE SET heartbeat_seconds = excluded.heartbeat_seconds",
        (ctx.agent.name, s),
    )
    return f"heartbeat set to {s}s for {ctx.agent.name}"


def _add_cron(
    *, id: str, expr: str, prompt: str,
    project_id: str | None = None,
    max_tokens: int | None = None,
) -> str:
    # Validate cron expression
    try:
        from croniter import croniter

        croniter(expr)
    except Exception as e:  # noqa: BLE001
        raise ToolError(f"invalid cron expression {expr!r}: {e}")
    ctx = current_context()
    if project_id is not None:
        from . import projects as _projects

        p = _projects.get(ctx.conn, project_id)
        if p is None or p.deleted_at is not None:
            raise ToolError(
                f"unknown project {project_id!r}. Call list_projects to see "
                "available projects, or get_current_session_info to use the "
                "project of the current session."
            )
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
        raise ToolError("max_tokens must be a positive integer if set")
    ctx.conn.execute(
        "INSERT INTO crons(agent_name, id, expr, prompt, enabled, project_id, max_tokens) "
        "VALUES (?,?,?,?,1,?,?) "
        "ON CONFLICT(agent_name, id) DO UPDATE SET "
        "expr=excluded.expr, prompt=excluded.prompt, enabled=1, "
        "project_id=excluded.project_id, max_tokens=excluded.max_tokens",
        (ctx.agent.name, id, expr, prompt, project_id, max_tokens),
    )
    proj_suffix = f" (project={project_id})" if project_id else ""
    budget_suffix = f" (max_tokens={max_tokens})" if max_tokens else ""
    return f"cron '{id}' set: {expr}{proj_suffix}{budget_suffix}"


def _remove_cron(*, id: str) -> str:
    ctx = current_context()
    cur = ctx.conn.execute(
        "DELETE FROM crons WHERE agent_name = ? AND id = ?", (ctx.agent.name, id)
    )
    if cur.rowcount == 0:
        raise ToolError(f"no cron with id '{id}'")
    return f"cron '{id}' removed"


def _list_crons() -> str:
    from . import projects as _projects

    ctx = current_context()
    rows = ctx.conn.execute(
        "SELECT id, expr, prompt, project_id, max_tokens "
        "FROM crons WHERE agent_name = ? AND enabled = 1 ORDER BY id",
        (ctx.agent.name,),
    ).fetchall()
    if not rows:
        return "(no crons)"
    lines = []
    for r in rows:
        proj_label = ""
        if r["project_id"]:
            p = _projects.get(ctx.conn, r["project_id"])
            if p is not None and p.deleted_at is None:
                proj_label = f" [project={p.name}]"
            else:
                proj_label = f" [project={r['project_id']} DELETED]"
        budget_label = f" [max_tokens={r['max_tokens']}]" if r["max_tokens"] else ""
        lines.append(
            f"{r['id']}: {r['expr']}{proj_label}{budget_label} — {r['prompt'][:60]}"
        )
    return "\n".join(lines)


_register(
    ToolSchema(
        name="set_heartbeat",
        description="Set this agent's heartbeat interval in seconds. Pass null or 0 to disable.",
        input_schema={
            "type": "object",
            "properties": {"seconds": {"type": ["number", "null"]}},
            "required": ["seconds"],
        },
    ),
    _set_heartbeat,
)

_register(
    ToolSchema(
        name="add_cron",
        description=(
            "Create or update a cron entry for this agent. `expr` is a 5-field "
            "UNIX cron expression; `prompt` is the starting message when the "
            "cron fires. Optional `project_id` binds each fire's session to a "
            "project (use `list_projects` to see available projects, or "
            "`get_current_session_info` for the current session's project). "
            "Omit or pass null for a project-less cron. Bound sessions inherit "
            "the project's system-prompt stanza from turn 1 and their uploads "
            "land in the project's uploads dir. Optional `max_tokens` is the "
            "per-turn token budget (cumulative input+output across iterations) "
            "for each fire; overrides the agent's setting and the global "
            "default. Use when this specific cron needs a larger or smaller "
            "budget than the agent's default."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "expr": {"type": "string"},
                "prompt": {"type": "string"},
                "project_id": {
                    "type": ["string", "null"],
                    "description": "Optional project uuid to bind each fire to.",
                },
                "max_tokens": {
                    "type": ["integer", "null"],
                    "description": (
                        "Optional per-turn token budget for each fire "
                        "(cumulative input+output across iterations)."
                    ),
                },
            },
            "required": ["id", "expr", "prompt"],
        },
    ),
    _add_cron,
)

_register(
    ToolSchema(
        name="remove_cron",
        description="Remove a cron entry by id.",
        input_schema={
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    ),
    _remove_cron,
)

_register(
    ToolSchema(
        name="list_crons",
        description="List this agent's cron entries.",
        input_schema={"type": "object", "properties": {}},
    ),
    _list_crons,
)


# ---------------------------------------------------------------------------
# Cross-session messaging
# ---------------------------------------------------------------------------


def _list_my_sessions(*, kind: str | None = None, limit: int = 20) -> str:
    from . import runtime  # local import to avoid cycle

    ctx = current_context()
    rows = runtime.list_sessions(ctx.conn, ctx.agent.name, kind=kind, limit=int(limit))
    if not rows:
        return "(no sessions)"
    out = []
    for r in rows:
        out.append(f"{r['id']}  {r['kind']:14s}  created_at={r['created_at']}")
    return "\n".join(out)


def _post_to_session(*, session_id: str, body: str) -> str:
    from . import broker, runtime
    from .types import AssistantText

    ctx = current_context()
    if not runtime.session_exists(ctx.conn, session_id, ctx.agent.name):
        raise ToolError(
            f"session {session_id} not found, or not owned by agent {ctx.agent.name}"
        )
    msg = AssistantText(text=body, injected_from=ctx.session_id)
    runtime.append_message(ctx.conn, session_id, msg)
    broker.publish(
        session_id,
        {
            "type": "injected_message",
            "session_id": session_id,          # target session (where this lands)
            "agent_name": ctx.agent.name,
            "from_session_id": ctx.session_id,  # source session (where this came from)
            "text": body,
        },
    )
    return f"posted to session {session_id}"


_register(
    ToolSchema(
        name="list_my_sessions",
        description="List sessions owned by the current agent, most recent first.",
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "description": "Optional filter: 'conversational', 'heartbeat', or 'cron'."},
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    _list_my_sessions,
)

_register(
    ToolSchema(
        name="post_to_session",
        description="Inject a message into another session owned by the same agent. The message is recorded as an assistant turn in the target session's history. Use list_my_sessions to find target session ids.",
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["session_id", "body"],
        },
    ),
    _post_to_session,
)


# ---------------------------------------------------------------------------
# Self-introspection: current time + current session info
# ---------------------------------------------------------------------------


def _get_current_time() -> dict:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return {
        "iso": now.isoformat(),
        "unix_ms": int(now.timestamp() * 1000),
        "weekday": now.strftime("%A"),
        "tz": "UTC",
    }


def _get_current_session_info() -> dict:
    import json as _json

    from . import models, runtime

    ctx = current_context()
    row = ctx.conn.execute(
        "SELECT kind, created_at, project_id FROM sessions WHERE id = ?",
        (ctx.session_id,),
    ).fetchone()
    if row is None:
        raise ToolError(f"session {ctx.session_id} not found")
    message_count = ctx.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        (ctx.session_id,),
    ).fetchone()[0]
    # Latest TurnMetrics row is the best proxy for "current context fill" —
    # input_tokens is what the model saw on the most recent call.
    last_metrics = ctx.conn.execute(
        "SELECT content_json FROM messages "
        "WHERE session_id = ? AND role = 'turn_metrics' "
        "ORDER BY id DESC LIMIT 1",
        (ctx.session_id,),
    ).fetchone()
    last_input_tokens = None
    last_output_tokens = None
    if last_metrics is not None:
        m = _json.loads(last_metrics["content_json"])
        last_input_tokens = m.get("input_tokens")
        last_output_tokens = m.get("output_tokens")
    context_window = models.context_window_for(
        ctx.agent.model, ctx.agent.max_context_tokens
    )
    project = runtime.session_project(ctx.conn, ctx.session_id)
    return {
        "session_id": ctx.session_id,
        "agent_name": ctx.agent.name,
        "model": ctx.agent.model,
        "kind": row["kind"],
        "created_at": row["created_at"],
        "message_count": message_count,
        "last_input_tokens": last_input_tokens,
        "last_output_tokens": last_output_tokens,
        "context_window": context_window,
        # null when the session isn't bound to a project
        "project_id": project.id if project is not None else None,
        "project_name": project.name if project is not None else None,
        "project_root": project.root if project is not None else None,
    }


def _get_project_info() -> dict | None:
    """Return the project record for the current session, or None if the
    session isn't in a project."""
    from . import runtime

    ctx = current_context()
    project = runtime.session_project(ctx.conn, ctx.session_id)
    if project is None:
        return None
    return {
        "id": project.id,
        "name": project.name,
        "root": project.root,
        "description": project.description,
        "project_context": project.project_context,
        "created_at": project.created_at,
    }


def _list_projects() -> list[dict]:
    """List active projects on the server."""
    from . import projects as _projects

    ctx = current_context()
    return [
        {
            "id": p.id,
            "name": p.name,
            "root": p.root,
            "description": p.description,
        }
        for p in _projects.list_projects(ctx.conn)
    ]


_register(
    ToolSchema(
        name="get_current_time",
        description=(
            "Return the current wall-clock time. Use this whenever you need to "
            "know the actual date or time — your training data has a cutoff and "
            "may not reflect today. Returns an object with `iso` (ISO 8601 UTC), "
            "`unix_ms`, `weekday`, and `tz` fields."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    _get_current_time,
)

_register(
    ToolSchema(
        name="get_current_session_info",
        description=(
            "Return metadata about the current session: session_id, agent name, "
            "model, session kind (`conversational`/`heartbeat`/`cron`), creation "
            "time, message count, the most recent input/output token counts "
            "alongside the model's context_window, and the project this session "
            "is bound to (`project_id`/`project_name`/`project_root`, or null "
            "for non-project sessions). Useful for self-monitoring (context "
            "fill) and for knowing which project root to operate against."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    _get_current_session_info,
)

_register(
    ToolSchema(
        name="get_project_info",
        description=(
            "Return the project record for the current session, or null if "
            "this session isn't bound to a project. Includes id, name, root "
            "path, description, and the project's context (system-prompt "
            "extension). Use this when you need richer project metadata than "
            "`get_current_session_info` exposes."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    _get_project_info,
)

_register(
    ToolSchema(
        name="list_projects",
        description=(
            "List all active (non-deleted) projects on the server, with id, "
            "name, root path, and description. Useful for finding a project's "
            "id when the user names one you're not currently in (e.g. to bind "
            "a cron to it via `add_cron(project_id=...)`)."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    _list_projects,
)


# ---------------------------------------------------------------------------
# File transfer to/from the client
# ---------------------------------------------------------------------------


def _list_uploads() -> str:
    from pathlib import Path

    from . import runtime, workspace as ws

    ctx = current_context()
    # Uploads land in the project root when the session is bound to a project;
    # otherwise in the agent's workspace. Mirror that here.
    proj = runtime.session_project(ctx.conn, ctx.session_id)
    base = Path(proj.root) if proj is not None else ctx.agent.workspace
    uploads = ws.uploads_dir(base)
    if not uploads.is_dir():
        return "(no uploads)"
    entries = sorted(
        (p for p in uploads.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not entries:
        return "(no uploads)"
    lines = []
    for p in entries:
        stat = p.stat()
        lines.append(f"uploads/{p.name}  ({stat.st_size} bytes)")
    return "\n".join(lines)


def _share_with_client(*, path: str, description: str = "") -> str:
    from . import broker, runtime, workspace as ws
    from .types import SharedFile

    ctx = current_context()
    try:
        full = ws.resolve(ctx.agent.workspace, path)
    except ws.WorkspaceError as e:
        raise ToolError(str(e))
    if not full.is_file():
        raise ToolError(f"not a file: {path}")

    rel = ws.relative_to_workspace(ctx.agent.workspace, full)
    size = full.stat().st_size
    runtime.append_message(
        ctx.conn, ctx.session_id, SharedFile(path=rel, description=description, size=size)
    )
    broker.publish(
        ctx.session_id,
        {
            "type": "file_available",
            "session_id": ctx.session_id,
            "agent_name": ctx.agent.name,
            "path": rel,
            "description": description,
            "size": size,
        },
    )
    return f"shared {rel} with the client ({size} bytes)"


_register(
    ToolSchema(
        name="list_uploads",
        description="List files the user has uploaded into this agent's workspace, newest first. Uploads land in `uploads/<filename>`. If two files were uploaded with the same name, the newer one is suffixed (e.g. `report-2.pdf`).",
        input_schema={"type": "object", "properties": {}},
    ),
    _list_uploads,
)

_register(
    ToolSchema(
        name="share_with_client",
        description="Make a file in this agent's workspace available to the user. The file appears in their UI as a downloadable artifact. `path` is workspace-relative; `description` is an optional short note shown alongside.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["path"],
        },
    ),
    _share_with_client,
)


# ---------------------------------------------------------------------------
# Skill meta-tools
# ---------------------------------------------------------------------------


def _list_skills() -> str:
    from . import mcp, skills

    ctx = current_context()
    always_loaded = set(ctx.agent.always_loaded_skills) | set(
        ctx.agent.always_loaded_mcp_servers
    )
    loaded = always_loaded | ctx.loaded_skills

    lines: list[str] = []
    # Python skills first.
    for name, desc in skills.manifest():
        marker = " [loaded]" if name in loaded else ""
        lines.append(f"{name}{marker} — {desc}")
    # MCP servers this agent may access.
    manager = mcp.get_manager()
    for server_name in ctx.agent.mcp_servers:
        srv = manager.get(server_name)
        if srv is None:
            lines.append(f"{server_name} (mcp) — not started")
            continue
        marker = " [loaded]" if server_name in loaded else ""
        status = "" if srv.error is None else f" [unavailable: {srv.error}]"
        desc = srv.description or f"{len(srv.tools)} tools"
        lines.append(f"{server_name} (mcp){marker}{status} — {desc}")

    if not lines:
        return "(no skills installed)"
    return "\n".join(lines)


def _load_skill(*, name: str) -> str:
    """Load a Python skill *or* an MCP server (unified surface)."""
    from . import mcp, skills

    ctx = current_context()
    skill = skills.get(name)
    if skill is not None:
        ctx.loaded_skills.add(name)
        tool_names = ", ".join(t.schema.name for t in skill.tools)
        return f"loaded skill '{name}'. tools available: {tool_names or '(none)'}"

    # Try MCP.
    if name in ctx.agent.mcp_servers:
        srv = mcp.get_manager().get(name)
        if srv is None:
            raise ToolError(f"MCP server {name!r} is configured but not started")
        if srv.error is not None:
            raise ToolError(f"MCP server {name!r} is unavailable: {srv.error}")
        ctx.loaded_skills.add(name)
        tool_names = ", ".join(t.name for t in srv.tools)
        return f"loaded MCP server '{name}'. tools available: {tool_names or '(none)'}"

    raise ToolError(f"unknown skill: {name}")


_register(
    ToolSchema(
        name="list_skills",
        description="List skills available to this agent. Each skill is a bundle of tools that can be loaded on demand.",
        input_schema={"type": "object", "properties": {}},
    ),
    _list_skills,
)

_register(
    ToolSchema(
        name="load_skill",
        description="Load a skill, making its tools available for the rest of this session.",
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    ),
    _load_skill,
)


# ---------------------------------------------------------------------------
# Active-tool resolution + execution
# ---------------------------------------------------------------------------


def active_schemas(agent: "AgentConfig", loaded_skills: set[str]) -> list[ToolSchema]:
    from . import mcp, skills

    schemas = [t.schema for t in BUILTINS.values()]

    # Python skills — always-loaded + session-loaded.
    for skill_name in list(agent.always_loaded_skills) + sorted(loaded_skills):
        skill = skills.get(skill_name)
        if skill is None:
            continue
        schemas.extend(t.schema for t in skill.tools)

    # MCP servers — always-loaded + session-loaded (session-loaded may include
    # either Python skills OR MCP server names since load_skill unifies them).
    manager = mcp.get_manager()
    mcp_active = set(agent.always_loaded_mcp_servers) | (
        loaded_skills & set(agent.mcp_servers)
    )
    for server_name in mcp_active:
        srv = manager.get(server_name)
        if srv is None or srv.error is not None:
            continue
        schemas.extend(srv.tools)

    # Deduplicate by name, preserving first occurrence.
    seen: set[str] = set()
    out: list[ToolSchema] = []
    for s in schemas:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def _resolve(name: str, agent: "AgentConfig", loaded_skills: set[str]):
    from . import skills

    if name in BUILTINS:
        return BUILTINS[name].fn, BUILTINS[name].is_async
    for skill_name in list(agent.always_loaded_skills) + sorted(loaded_skills):
        skill = skills.get(skill_name)
        if skill is None:
            continue
        for t in skill.tools:
            if t.schema.name == name:
                return t.fn, t.is_async
    return None, False


def _resolve_mcp(
    name: str, agent: "AgentConfig", loaded_skills: set[str]
) -> tuple[str, str] | None:
    """If `name` is an active MCP-backed tool for this agent, return
    (server_name, raw_tool_name). Otherwise None."""
    from . import mcp

    if NAMESPACE_SEP not in name:
        return None
    server_name, _ = name.split(NAMESPACE_SEP, 1)
    if server_name not in agent.mcp_servers:
        return None
    active = set(agent.always_loaded_mcp_servers) | (
        loaded_skills & set(agent.mcp_servers)
    )
    if server_name not in active:
        return None
    srv = mcp.get_manager().get(server_name)
    if srv is None or srv.error is not None:
        return None
    raw = srv.raw_tool_names.get(name)
    if raw is None:
        return None
    return server_name, raw


NAMESPACE_SEP = "__"


def schemas() -> list[ToolSchema]:
    """Backwards-compatible: schemas of the built-ins only."""
    return [t.schema for t in BUILTINS.values()]


async def execute(name: str, args: dict[str, Any], *, ctx: ToolContext) -> tuple[str, bool]:
    # MCP-backed tools take a different path — no cwd change, no contextvar
    # (external process; no notion of Ark's current session).
    mcp_target = _resolve_mcp(name, ctx.agent, ctx.loaded_skills)
    if mcp_target is not None:
        from . import mcp as mcp_module

        server_name, raw_name = mcp_target
        cfg = ctx.config.mcp_servers.get(server_name)
        timeout = cfg.timeout_seconds if cfg is not None else 30.0
        try:
            output = await mcp_module.get_manager().call_tool(
                server_name, raw_name, args, timeout=timeout
            )
            return output, False
        except mcp_module.MCPError as e:
            return str(e), True

    fn, is_async = _resolve(name, ctx.agent, ctx.loaded_skills)
    if fn is None:
        return f"unknown tool: {name}", True

    def runner() -> str:
        import os

        token = _context.set(ctx)
        prev = os.getcwd()
        try:
            os.chdir(ctx.cwd)
        except FileNotFoundError:
            ctx.cwd.mkdir(parents=True, exist_ok=True)
            os.chdir(ctx.cwd)
        try:
            result = fn(**args)
        finally:
            os.chdir(prev)
            _context.reset(token)
        return _to_str(result)

    async def async_runner() -> str:
        token = _context.set(ctx)
        try:
            result = await fn(**args)
        finally:
            _context.reset(token)
        return _to_str(result)

    try:
        if is_async:
            output = await async_runner()
        else:
            output = await asyncio.to_thread(runner)
        return output, False
    except ToolError as e:
        return str(e), True
    except TypeError as e:
        return f"invalid arguments: {e}", True
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}", True


def _to_str(result: Any) -> str:
    if isinstance(result, str):
        return result
    if result is None:
        return ""
    import json as _json

    try:
        return _json.dumps(result)
    except (TypeError, ValueError):
        return repr(result)
