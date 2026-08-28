# Changelog

## Unreleased — Context efficiency: named context blocks, tool-result elision, prompt caching, /context endpoint

Four related changes that stop a long session's prompt from growing
without bound and make its cost observable:

- **Named context blocks.** `POST /agents/{name}/sessions/{sid}/context`
  accepts an optional `name`; appending the same name again REPLACES the
  block in the system prompt (last text, first position) instead of
  growing it forever. Unnamed appends keep the additive behavior. The
  response gains `replaced: true|false`. Rows are append-only — the
  dedupe happens at prompt-build time.
- **Tool-result elision (in-memory).** Once the in-view tool-result bytes
  exceed `tool_result_max_bytes` (default 64 KB), ONE pass replaces
  results older than the last `tool_result_keep_turns` (2) user turns and
  larger than `tool_result_elide_over` (2 KB) with a short stub naming
  the tool and call id. Hysteresis by design — a per-turn trim would bust
  the provider prompt cache. Rows are never touched.
- **Prompt caching + cache telemetry.** `agents.<name>.prompt_caching`
  (default true) adds `cache_control` markers on Anthropic (native and
  `anthropic/*` via OpenRouter). `TurnMetrics`/`turn_usage` gain
  `cached_input_tokens` / `cache_write_tokens`, with `input_tokens`
  normalized to the TOTAL prompt on every provider path
  (OpenAI-compatible `prompt_tokens_details.cached_tokens`, Google
  `cached_content_token_count`). The CLI usage line shows `% cached`.
- **`GET /agents/{name}/sessions/{sid}/context`** reports the live system
  prompt size, every context block (named blocks deduped exactly as
  rendered), the latest usage numbers and the compaction count.

New agent config knobs: `prompt_caching`, `tool_result_keep_turns`,
`tool_result_elide_over`, `tool_result_max_bytes`. `ark/models.py` adds
the `kimi-k3` context window (256k).


## Unreleased — Cron entries can be bound to a project

A cron entry can now carry an optional `project_id`. Every fire of that
cron creates a session already attached to the project — system prompt
gets the project stanza from turn 1, uploads land in the project's dir.
See [docs/projects.md § Cron entries can be bound to a project](docs/projects.md#cron-entries-can-be-bound-to-a-project).

### Schema migration v5

Adds `crons.project_id TEXT` (nullable). Existing crons keep firing
project-less sessions unchanged. No FK — a cron intentionally survives a
soft-deleted project (the scheduler warns + fires anyway; the resulting
session runs project-less).

### REST changes

```
PUT /agents/{name}/crons/{cron_id}
Body: { "expr": "...", "prompt": "...", "project_id"?: "<uuid>" | null }
```

- `project_id` present with a string → validated (404 on unknown or
  soft-deleted target), then bound.
- `project_id: null` → detach.
- `project_id` omitted → **preserve existing binding on update** (so
  "just change the schedule" doesn't accidentally clobber the project
  binding), null on insert.

`GET /agents/{name}/crons` and `GET /agents/{name}` return `project_id`
and (on the former) `project_name` for each cron. A cron whose bound
project was soft-deleted returns `project_id` unchanged with
`project_name: null`.

### Agent tools

- `add_cron(id, expr, prompt, project_id?)` — new optional parameter,
  validated at add time. Backwards-compatible: existing 3-arg calls keep
  working.
- `list_crons` now shows `[project=<name>]` (or `[project=<id> DELETED]`
  for dangling refs) next to each cron.
- **New `list_projects` tool** — returns the active (non-deleted)
  projects with `id`, `name`, `root`, `description`. Complements
  `get_current_session_info` (which only surfaces the current session's
  project) — useful when the user names a project the agent isn't
  currently in.

### Scheduler

- Reads `crons.project_id` on tick, threads it through `_fire_cron` →
  `_drive` → `runtime.create_session`.
- If the bound project is soft-deleted at fire time: logs a warning to
  stderr (`[scheduler] cron X for agent Y: bound project Z is deleted,
  firing in workspace mode`), still fires. The session row records the
  dangling id (audit trail); `runtime.session_project()` returns None for
  the soft-deleted project so the session runs project-less.

### CLI

```
ark cron set <agent> <id> "<expr>" --prompt "..." --project <name>    # bind by name
ark cron set <agent> <id> "<expr>" --prompt "..." --no-project         # detach
ark cron set <agent> <id> "<expr>" --prompt "..."                      # keep whatever's bound
ark cron list <agent>                                                   # now shows [project=<name>]
```

`--project` accepts a name (resolved via `GET /projects`);
`--no-project` and `--project` are mutually exclusive.

### Sharp edges

- **No cascade on project delete.** A cron bound to a soft-deleted
  project keeps firing project-less. This is intentional — the user might
  restore the project (rename another to it, etc.) and expects the cron
  binding to remain. If it's undesired, remove the cron or PATCH it
  detach.
- **No connection between cron-fired sessions and a user's conversational
  session** in the same project. Use `post_to_session` if the cron needs
  to surface output to a human's active thread.
- **Heartbeats are unchanged** — they're agent-level, not project-level.
  A heartbeat that needs project scope can name the project root
  explicitly in its `heartbeat_prompt.md`.

## Unreleased — Mutable session ↔ project binding

Session-to-project assignment was previously immutable. It's now mutable
via a dedicated endpoint, and the LLM is explicitly notified of the
transition on the next turn so it doesn't silently start seeing a
different project's environment.

### New REST endpoint

```
PATCH /agents/{name}/sessions/{sid}/project
Body: { "project_id": "<uuid>" }   # reassign or first-time assign
      { "project_id": null }         # detach

200: { "ok": true, "changed": true, "from": {id,name,root}|null, "to": {id,name,root}|null }
200: { "ok": true, "changed": false }        # no-op (already assigned as requested)
404: unknown agent, session, or project (soft-deleted target counts as unknown)
409: session has unmatched tool calls
400: body missing 'project_id', or wrong type
```

Idempotent: PATCHing to the current binding returns `{"changed": false}`
without writing a marker or publishing an event.

### New message kind: `ProjectAssignmentChanged`

Persisted in history on every real change (skipped on no-ops). Fields:
`from_project_id`, `to_project_id`, `from_project_name`, `to_project_name`,
`from_root`, `to_root`, `changed_at`. Both endpoints can be null (detach /
first-time-assign). `GET /history` returns it so clients can render a
timeline divider; the runtime substitutes it with a synthetic `UserText`
notification when building the LLM's message list so the model sees the
transition as an event at that point in the conversation (previous
project → new project + a note that prior file references are
historical).

### New WS event

`session_project_changed` on `/events`:

```json
{
  "type": "session_project_changed",
  "session_id": "...", "agent_name": "...",
  "from_project_id": "...", "from_project_name": "...",
  "to_project_id": "...", "to_project_name": "...",
  "changed_at": 1755600000000
}
```

Only fires on real changes — no-op PATCHes are silent.

### New CLI

```
ark session set-project <sid> <project-name>     # reassign
ark session set-project <sid> --none              # detach
```

Auto-resolves the session's owning agent from `GET /sessions/{sid}` so the
user doesn't have to specify `--agent`.

### Runtime changes

- New helper `runtime.set_session_project(conn, sid, new_project_id)` —
  updates `sessions.project_id` and appends the marker in one call.
  Returns `(from_project, to_project)` or `None` for no-op.
- New helper `runtime._rewrite_for_llm(messages)` — pre-provider
  substitution pass; currently only rewrites `ProjectAssignmentChanged`
  markers to their `UserText` notification form. `run_user_turn` and
  `compact_session` both use it.
- No DB schema change (`content_json` handles the new kind natively).

### Sharp edges

- **Old uploads become invisible via `list_uploads`** after reassignment
  — files still exist under the old project's `uploads/`, but the current
  session's tool sees only the new project's dir. The transition
  notification explicitly warns about historical references.
- **Compaction across a reassignment** relies on the summarizer preserving
  the transition. The default prompt asks for that; if it drops in
  practice, clients can pre-supply a summary via `POST .../compact`.
- **Per-agent access control isn't added here** — any client with the
  bearer token can reassign any session to any project. If per-agent /
  per-user gating is needed, that's a separate authz layer.

### Docs

- [docs/projects.md](docs/projects.md) — new "Reassigning a session's
  project" section. Removed the "binding is immutable" language.
- [docs/sessions.md](docs/sessions.md) — event table + history kinds
  updated.

## Unreleased — Manual session compaction

Client-invoked companion to automatic compaction. Same underlying mechanism
and events; new REST + CLI surface for on-demand triggering.

### New REST endpoint

```
POST /agents/{name}/sessions/{sid}/compact
Body: {}                        # server-generated summary
      { "summary": "..." }      # client-supplied text, no LLM call

200: { "ok": true, "summary": "<text>", "reason": "client-invoked" | "client-supplied" }
502: { "ok": false, "code": "<classified>", "message": "..." }
```

- `404` for unknown agent or session.
- `409` if the session is mid-tool-loop (any `ToolCall` without a matching
  `ToolResult`) — same sharp-edge as the reactive trigger; compacting
  across that boundary would orphan a `ToolResult`.
- `400` if `summary` is provided but empty or non-string.
- Fires `compaction_started` → `compaction_completed`/`_failed` on
  `/events` so connected WS clients see the work.
- **Ignores `compaction_enabled`** on the agent — that flag only gates
  the automatic triggers; explicit client requests always run.

### New CLI slash command

Mid-chat:

```
you> /compact                                # server-generated
you> /compact set: <your summary text>       # supplied
```

### Client rendering

`_handle_event` in the CLI now renders the four compaction event types
(`compaction_started`, `_completed`, `_failed`, `_skipped`) with a
"Compacting session (N% full)" status line and a summary-length ack. All
three trigger paths (proactive/reactive/client-invoked) surface
identically to any client subscribed to `/events`.

## Unreleased — Automatic session compaction

Sessions that approach the model's context window are now automatically
summarized. Prior turns get folded into a `CompactionSummary` message
(persisted, visible in history), and subsequent turns see only the
summary + post-compaction turns. See
[docs/sessions.md § Compaction](docs/sessions.md#compaction) for the
reference.

### Mechanism

One new message type — `CompactionSummary(text, reason)` — persisted like
any other. Runtime rule: when building the LLM's message list, if any
`CompactionSummary` exists, take only messages after the LATEST one; fold
its text into the system prompt as a "Prior conversation (summarized)"
stanza. Older messages stay in history for audit/replay, invisible to
the LLM. No DB schema migration — `messages.content_json` handles the
new kind natively.

### Triggers

Two triggers, both automatic:

- **Proactive**: at turn start, if last observed
  `TurnMetrics.input_tokens ≥ compaction_threshold × context_window`,
  compact before persisting the incoming user message. The user message
  becomes the first post-compaction turn.
- **Reactive**: on `context_too_long` at the first iteration of a turn,
  compact and retry the same turn once. Reactive only runs at turn start
  (last message is `UserText`) — mid-tool-loop failures fall through to
  the existing error path.

Compaction attempts at most once per turn. Reactive after successful
proactive is not attempted.

### Config

Per-agent, both optional (defaults `true` / `0.85`):

```json
"agents": {
  "scribe": {
    ...
    "compaction_enabled": true,
    "compaction_threshold": 0.85
  }
}
```

Existing configs pick up the defaults without change.

### New events on `/events`

Four events for full client traceability of compaction work:

| Event | When |
|---|---|
| `compaction_started` | About to run the summarizer (`reason`, `input_tokens`, `context_window`, `model`) |
| `compaction_completed` | Summary persisted (`summary`, `reason`) |
| `compaction_failed` | Summarizer errored (`code`, `message`, `reason`) — turn proceeds/falls through per trigger type |
| `compaction_skipped` | Threshold crossed but `compaction_enabled: false` — warning-only |

Client UX: render "Compacting session… (context was 87% full)" between
`_started` and `_completed`, and show the resulting summary alongside a
visual divider in the transcript. Everything before the divider can be
rendered collapsed-by-default so the user can still scroll back.

### Summarizer

Uses the session's own provider + model. Prompt asks the model to preserve
names, facts, decisions, files (by path), code discussed or written,
commitments, open questions, and significant tool results. Omits
persona/environment (provided separately). If a prior `CompactionSummary`
exists, its text is passed as background so information isn't lost across
successive compactions.

### Sharp edges

- **Once per turn.** No infinite compact-retry loops.
- **6-message floor** since the last compaction — no point summarizing a
  handful of turns.
- **Post-compaction fill is unknown** until the next turn's `TurnMetrics`
  lands; proactive check detects this via a "metrics predate latest
  compaction" guard and skips.
- **Reactive is idle-only** — mid-tool-loop context overflow still fails
  fast rather than compacting across an unmatched `ToolCall`/`ToolResult`
  boundary.
- **Summarizer quality is load-bearing**; old summaries persist in
  history as an audit trail. Restore-from-prior-summary is a natural
  future addition but not shipped in this cut.
- **Cost**: one extra provider call per compaction. Model override for
  the summarizer (cheap tier) is a natural future config knob.

### Client migration

Additive — existing clients continue to work. To surface the feature:

1. Handle the four new event types (`compaction_started`,
   `_completed`, `_failed`, `_skipped`). At minimum, show a spinner while
   between started and completed.
2. Handle `CompactionSummary` in `GET /history` — render as a divider
   with the summary body expandable, and consider collapsing everything
   older.
3. Nothing else changes — the underlying turn/error/tool events stream
   identically.

## Unreleased — MCP servers as first-class tool sources

Ark speaks [Model Context Protocol](https://modelcontextprotocol.io) as a
client. Configured MCP servers appear to agents alongside Python skills,
same discovery + loading affordances. See [docs/mcp.md](docs/mcp.md) for
the full reference.

### New config

Two additive blocks, both optional:

```json
"mcp_servers": {
  "linear":   { "transport": "stdio", "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-linear"],
                "env": { "LINEAR_API_KEY": "lin_..." } },
  "notion":   { "transport": "http", "url": "https://mcp.notion.com/mcp",
                "headers": { "Authorization": "Bearer nti_..." } }
},
"agents": {
  "scribe": {
    ...
    "mcp_servers": ["linear", "notion"],           // per-agent whitelist
    "always_loaded_mcp_servers": ["linear"]        // schemas exposed every turn
  }
}
```

Existing configs continue to work unchanged.

### Tool namespace

MCP tool names are prefixed with the server name: `linear__create_issue`,
`postgres__query`. Double-underscore separator so no provider's
tool-schema validator objects.

### Unified with skills

`list_skills` shows both Python skills and MCP servers (tagged `(mcp)`),
and `load_skill("linear")` works uniformly — the distinction is invisible
from the agent's perspective. The only difference is the tool-name
prefix.

### Lifecycle

Persistent connections opened at server boot and reused across all
sessions. Per-server startup failures don't abort Ark — the server is
marked unavailable, its tools return errors when called. Connections
close cleanly on shutdown.

`GET /agents/{name}` now includes per-agent MCP server status
(`ready`/`error`/`tool_count`/`always_loaded`) so clients can render an
"MCP health" panel.

### Sharp edges

- **stdio servers are uvicorn subprocesses**. Clean shutdown via
  `systemctl restart`; `kill -9` may leave zombies.
- **MCP tools have no context**. No `current_context()`, no DB, no
  workspace. Right boundary for external integrations.
- **Token bloat if abused**. Always-loading multiple 50-tool servers is
  real cost. Prefer lazy `load_skill` unless the agent uses those tools
  every turn.
- **No per-tool ACL yet**. You can gate at the server level, not per
  tool. Comes later.

### Dependency

Adds `mcp>=1.0` (official Python SDK). Requires Python 3.10+; the
production container is on 3.11 and the dev container on 3.12, so this is
fine. The local dev venv on macOS Python 3.9 keeps working — the SDK
import in `ark/mcp.py` is lazy and gated on config, and MCP-specific
tests use a stub connection factory.

## Unreleased — Cron fire history + session metadata + ark show

For debugging "what did this cron actually do," the scheduler now records
which cron entry triggered each fire, and there are dedicated endpoints +
CLI commands for inspecting the history.

### Schema migration v4

Adds `sessions.cron_id TEXT` (nullable). Populated only for sessions
created by the scheduler firing that cron. Existing sessions stay `NULL`.
No backfill — history starts now.

### New REST endpoints

```
GET /sessions/{sid}
  → { id, agent_name, kind, created_at, ended_at, project_id, cron_id,
      cron_prompt? }   # cron_prompt present only when kind='cron'

GET /agents/{name}/crons/{cron_id}/sessions?limit=N
  → [ { session_id, created_at, ended_at, had_error, error_code, summary }, … ]
```

`summary` resolution order: first `post_to_session.body` (covers the most
common cron pattern — "send a briefing"), then last `AssistantText`, then
`"(no output)"` for fires that produced nothing (Gemini safety filter,
empty user input, etc.).

`had_error` + `error_code` come from any `RunError` row persisted during
the run.

### New CLI commands

```
ark cron history <agent> <cron-id> [--limit 20]
ark show <session-id>
```

`ark show` works for any session kind (cron, heartbeat, conversational).
It collapses turns into a readable transcript, surfaces `RunError` and
`TurnMetrics` rows inline, and prints the cron prompt when applicable.

## Unreleased — Recursive directory delete

`DELETE /projects/{id}/files/{path}` and `DELETE /agents/{name}/files/{path}`
now remove directories recursively (whole subtree). Previously they only
removed empty directories, returning `409` otherwise.

**Behavior change for clients**: if you were relying on `409` to detect
"this is a non-empty directory" and then prompting the user to confirm,
that signal is gone — the call now succeeds and the contents are removed.
If you want a confirm-before-recursive-delete UX, gate that on the client
side using the directory listing returned by `GET`.

Defense-in-depth note: the handler explicitly refuses to delete a path
that resolves to the project root or workspace root itself, even though
URL normalization already eats `.` / `..` segments before they reach the
handler.

## Unreleased — File rename

Adds an `op=rename` action to the file management `POST` handler on both
the project and workspace filesystem endpoints. Works on files and
directories, never silently overwrites, and applies path-traversal checks
to both source and destination.

```
POST /projects/{id}/files/{path}?op=rename&dest=<dest>
POST /agents/{name}/files/{path}?op=rename&dest=<dest>
```

Responses on success: `{"ok": true, "from": "<old>", "to": "<new>"}`.
Errors: `400` if `dest` is missing or escapes the root; `404` if the
source doesn't exist; `409` if `dest` already exists.

## Unreleased — Workspace filesystem REST + live events

Adds a browsable / editable REST surface for an agent's workspace,
mirroring the project filesystem endpoints. The previously download-only
`GET /agents/{name}/files/{path}` now also returns directory listings when
the target is a directory, and is joined by `PUT` / `DELETE` / `POST ?op=mkdir`
for symmetry with `/projects/{id}/files/...`. Every agent's workspace is
now also filesystem-watched, with changes fanning out as a new
`workspace_file_changed` event on `/events`.

### New on the wire

**REST**:

```
GET    /agents/{name}/files                  # list workspace root
GET    /agents/{name}/files/{path}           # file → bytes; dir → JSON listing
PUT    /agents/{name}/files/{path}           # write file (raw body)
DELETE /agents/{name}/files/{path}           # delete file or empty dir
POST   /agents/{name}/files/{path}?op=mkdir
```

**New WS event** on `/events`:

```json
{
  "type": "workspace_file_changed",
  "agent_name": "scribe",
  "path": "scratch/draft.md",
  "change": "created" | "modified" | "deleted"
}
```

Same coalescing window and ignore-list as `project_file_changed`.

### Behavior change worth flagging

The existing `GET /agents/{name}/files/{path}` endpoint **adds a new
behavior** when the target path is a directory: it now returns a JSON
listing instead of 404. Files continue to stream as bytes. If a client was
relying on directory-paths returning 404, switch to checking the response
content-type or shape.

### Server-side internals

- `ark/file_watcher.py` generalized: a `FileWatcher` now hosts multiple
  *subjects* (kinds: `project` or `workspace`) with one Observer.
  `watch(kind, id, root)` / `unwatch(kind, id)`. Event-type and id-field
  mapping is data-driven.
- `ark/server.py`: workspace endpoints added; lifespan now starts a
  workspace watch for every configured agent at boot.
- No DB schema change.

## Unreleased — Projects

Adds a new concept of **projects** — shared user-visible working
directories that one or more sessions can be bound to. Unlike an agent's
private workspace, a project's contents are intended for the user to
inspect, edit, upload to, and watch changing in real time. Multiple agents
can work in one project.

See [docs/projects.md](docs/projects.md) for the full reference.

### New on the wire

**REST: project CRUD**

```
POST   /projects                         # create
GET    /projects                         # list active; ?include_deleted=true for all
GET    /projects/{id}
PUT    /projects/{id}                    # update name/description/project_context
DELETE /projects/{id}                    # soft-delete (files survive)
```

**REST: per-project filesystem**

```
GET    /projects/{id}/files
GET    /projects/{id}/files/{path}       # file → bytes; dir → JSON listing
PUT    /projects/{id}/files/{path}       # raw body
DELETE /projects/{id}/files/{path}
POST   /projects/{id}/files/{path}?op=mkdir
```

**Session creation** (`POST /agents/{name}/sessions`) now accepts an optional
`project_id` to bind the session to a project at creation. Binding is
immutable for the life of the session.

```diff
  POST /agents/{name}/sessions
  {
    "context": "...",
+   "project_id": "<project-uuid>"
  }
```

**New WS event** on `/events`:

```json
{
  "type": "project_file_changed",
  "project_id": "<uuid>",
  "path": "subdir/draft.md",
  "change": "created" | "modified" | "deleted"
}
```

Coalesced within ~200ms, with a default ignore-list (`.git`,
`node_modules`, `__pycache__`, etc.).

### New / changed agent tools

| Tool | Change |
|---|---|
| `get_current_session_info` | Now includes `project_id`, `project_name`, `project_root` (null when the session isn't in a project). |
| `get_project_info` | **NEW.** Returns the project record (id, name, root, description, project_context) or null. |
| `list_uploads` | Now dispatches: lists `<project_root>/uploads/` when in a project, `<workspace>/uploads/` otherwise. No call-site change. |

### Behavior changes for project sessions

- **System prompt** gains a "Project (this session)" section between the
  Environment stanza and per-session context. The agent is told the project
  root path and instructed to default to operating under it unless
  explicitly asked to modify the workspace.
- **Uploads** land in `<project_root>/uploads/` instead of the workspace's
  uploads dir.
- **`cwd` is unchanged** — still the agent's workspace. The agent uses
  absolute paths when operating on project files. This was a deliberate
  call to keep `cwd` predictable across all sessions.

### DB schema

Migration to user_version 3 adds:

- New `projects` table (id, name, root, description, project_context,
  created_at, deleted_at).
- New `project_id` column on `sessions` (nullable, FK).
- Unique index on `projects(name)` filtered to non-deleted rows, so a
  deleted project's name can be reused.

Applied automatically on server start.

### Client migration notes

This release is **additive** — existing clients that don't use projects
continue to work unchanged. To adopt projects:

1. Surface a project picker. On startup, `GET /projects` for the list.
2. When creating a session, optionally include `project_id` in the body.
3. Handle the new WS event type `project_file_changed`. Filter by
   `project_id` if you only care about specific projects.
4. Build a file browser / editor against the per-project filesystem
   endpoints. Listings are JSON; file contents are raw bytes.

### Server-side internals

- New `ark/projects.py` for CRUD + path resolution (mirrors `ark/workspace.py`).
- New `ark/file_watcher.py` — `watchdog`-based per-project filesystem
  watcher, with coalescing + ignore-list, publishing to the broker.
- `ark/runtime.py`: `session_project()` helper; `system_prompt` accepts a
  `project` arg.
- `ark/server.py`: project CRUD + filesystem endpoints; lifespan starts the
  watcher and adds watches for all active projects.
- `watchdog>=4.0` added to `requirements.txt`.

## Earlier — Unified event stream

(Original entry — see git history for details.) Replaced the per-session
WebSocket with a single per-client event stream (`WS /events`), added a
cross-session catch-up REST endpoint (`GET /events?since_id=...`), and
removed the old per-session WS endpoint. Every server-pushed event now
carries `session_id` and `agent_name`; commands sent over the WS specify
their target session in the body.
