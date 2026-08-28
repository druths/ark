# Sessions

A *session* is one conversation thread with an agent. Three kinds exist
(`conversational`, `heartbeat`, `cron` — see [design/design.md](../design/design.md)),
all sharing the same shape: a UUID, an agent owner, an ordered list of
messages, and an open-ended lifetime.

This doc covers the session-level API surface. For file transfer (uploads,
downloads, agent-shared artifacts) see [files.md](files.md).

## REST endpoints

All require `Authorization: Bearer <auth_secret>`.

### Create a session

```
POST /agents/{name}/sessions
Body (optional): { "context": "..." }
→ { "id": "<uuid>" }
```

Always creates a `conversational` session. If `context` is provided, it's
appended as the session's first `SessionContext` message — see
[Per-session context](#per-session-context) below.

Empty or absent bodies are accepted (creates an empty session, no context).
`Content-Type` of `application/json` is not strictly required for the empty
case.

### List sessions

```
GET /agents/{name}/sessions
GET /agents/{name}/sessions?kind=conversational&limit=20
→ [ { id, agent_name, kind, created_at, ended_at }, ... ]
```

Most recent first. `kind` filters to `conversational`, `heartbeat`, or
`cron`. `limit` defaults to 50.

### Read history

```
GET /agents/{name}/sessions/{sid}/history
→ [ { "kind": "<MessageKind>", "data": {...} }, ... ]
```

Returns every message in the session, ordered chronologically. `kind` is the
class name of the message (`UserText`, `AssistantText`, `ToolCall`,
`ToolResult`, `UploadMessage`, `SharedFile`, `SessionContext`,
`CompactionSummary`, `ProjectAssignmentChanged`).

### Delete a session

```
DELETE /agents/{name}/sessions/{sid}
→ { "ok": true }
```

Deletes the session row and (via cascade) all of its messages. Workspace
files survive — they're agent-scoped, not session-scoped. See
[files.md](files.md) for the file lifecycle nuance.

## Per-session context

The agent's persona and behavior come from `<ARK_HOME>/agents/<name>/session_context.md`
(see [config.md](config.md)) — that's set by whoever runs the server and is
the same across all sessions of one agent. **Per-session context** lets a
*client* layer additional instructions on top, just for one session, without
touching the agent file.

### How it stacks

The system prompt sent to the model is composed top-to-bottom:

```
<agent's session_context.md content>          ← agent persona/identity

---
Environment (managed by the Ark harness, do not invent paths):
- Your name: ...
- Your workspace directory: ...
- ...                                          ← runtime facts

---
Session context (provided by the client for this session —
additive, do not override the agent context above):
<context message 1>

<context message 2>                            ← client-supplied,
                                                  only if any have been added
```

The "do not override" line makes the layering legible to the model.

### Adding context

```
POST /agents/{name}/sessions
Body: { "context": "..." }                    ← seeds on session creation

POST /agents/{name}/sessions/{sid}/context
Body: { "context": "..." }                    ← appends mid-session
→ { "ok": true, "count": N }                  ← N = total context messages
```

The mid-session endpoint:
- **Always appends.** No replace, no edit. Multiple posts accumulate in
  the order they arrive.
- Rejects empty/whitespace-only text with `400`.
- Returns the total context-message count so the client can confirm.

### Behavior

- **Visibility timing.** New context shows up on the *next* user turn, not
  within an in-flight one. The system prompt is built once at the start of
  each turn; mid-tool-loop additions wait their turn.
- **Not sent as LLM messages.** `SessionContext` rows live in the session
  history (for audit + replay) but the runtime strips them before passing
  the message list to the provider. They contribute to the system prompt
  only — otherwise the model would see the same text twice.
- **History inspection.** They appear as `{"kind": "SessionContext"}` in
  `GET /history` so clients can see what's been added.
- **No DELETE.** Wipe-and-restart isn't supported in v1. If you really need
  a clean slate, delete the session and create a new one.

### CLI

```
ark chat <agent> --context "..."             # seed at creation
ark chat <agent> --context-file PATH         # read from file
ark chat <agent> --session SID --context "..." # append to a resumed session

# mid-chat
you> /context <additional instructions>
```

### Named blocks

`POST .../context` accepts an optional `"name"`. A named block is
REPLACED in the system prompt when the same name is appended again — the
last text renders at the first occurrence's position, so a periodically
refreshed block (a focus object, a mode note) costs one slot instead of
growing the prompt every refresh. Unnamed appends stay additive. The
response reports `replaced: true|false`. Rows in history are append-only;
the dedupe is prompt-build-time only.

`GET /agents/{name}/sessions/{sid}/context` returns the introspection
view: `system_prompt_bytes`, `blocks[{name, bytes, seq}]` (deduped as
rendered), `last_input_tokens`, `last_cached_input_tokens`,
`context_window`, `compactions`.

### Tool-result elision

Old, oversized tool results are elided from the LLM's view (never from
history) once the in-view result bytes exceed
`tool_result_max_bytes` (64 KB): one pass replaces results older than the
last `tool_result_keep_turns` (2) user turns and larger than
`tool_result_elide_over` (2 KB) with
`[tool result elided: N bytes; tool <name>; call <id> — re-run if needed]`.
The single-pass hysteresis keeps the provider prompt cache warm.

## The event stream (unified per-client)

Ark exposes a **single WebSocket per client** that carries events for every
session the bearer token has access to. The same connection delivers
streaming text from the active chat, cron-injected messages from other
sessions, tool calls happening in background runs, and so on — clients
multiplex on `session_id` to decide what to render where.

```
WS /events
Authorization: Bearer <auth_secret>           # header, or ?token=... in URL
```

Every event the server pushes has `session_id` and (where applicable)
`agent_name`. Commands the client sends carry `session_id` to route to the
right session.

### Server → client events

| Event | Fields | When |
|---|---|---|
| `assistant_delta` | `text` | Streaming assistant text |
| `assistant_message` | `text` | End of one provider turn |
| `thinking` | `delta` | Extended-thinking text (Gemini/Anthropic) |
| `tool_call` | `id`, `name`, `input` | Model invoked a tool |
| `tool_result` | `id`, `output`, `error` | Tool returned |
| `turn_usage` | `input_tokens`, `output_tokens`, `model`, `context_window` | Per-turn token counts (`context_window` is null if unknown). See [Usage tracking](#usage-tracking) below. |
| `file_available` | `path`, `description`, `size` | Agent shared a file (see [files.md](files.md)) |
| `injected_message` | `from_session_id`, `text` | Another session injected a message via `post_to_session` |
| `error` | `code`, `message` | Classified failure. `code` is one of `context_too_long`, `rate_limit`, `auth`, `other`. The runtime persists the same error as a `RunError` message in history. |
| `compaction_started` | `reason`, `input_tokens`, `context_window`, `model` | Compaction is about to run (see [Compaction](#compaction)) |
| `compaction_completed` | `summary`, `reason` | Summary persisted; subsequent turns use it |
| `compaction_failed` | `code`, `message`, `reason` | Summarizer call errored |
| `compaction_skipped` | `reason`, `input_tokens`, `context_window` | Threshold crossed but compaction is disabled — warning-only, no action taken |
| `session_project_changed` | `from_project_id`, `from_project_name`, `to_project_id`, `to_project_name`, `changed_at` | A session's project binding changed (see [projects.md § Reassigning a session's project](projects.md#reassigning-a-sessions-project)) |
| `done` | `stop_reason` | Whole run-loop finished for that session, awaiting next user input. On classified errors, `stop_reason` is `"error:<code>"`. |

Every event also carries `session_id` and (except for the broad "error" case
where the session couldn't be identified) `agent_name`.

### Client → server commands

| Command | Required fields | Effect |
|---|---|---|
| `user_message` | `session_id`, `text` | Start a new turn in that session. Multiple sessions can have turns running concurrently — events stream back tagged with their `session_id`. |
| `stop` | `session_id` | Request cancellation (v1: no-op — see design.md §6). |

Per-session context is **not** added over the WS — it's a REST operation
even mid-chat. The CLI does the REST call when you type `/context ...`.

### Cross-session catch-up

```
GET /events?since_id=<int>&since_ms=<int>&limit=<int>
```

Returns persisted messages across *every* session, ordered by the monotonic
message id. Use this to fill the gap between disconnect and reconnect, to
compute unread counts, or to populate "what's new since I last opened the
app" UIs.

Response:

```json
{
  "events": [
    {
      "id": 12345,
      "session_id": "...",
      "agent_name": "scribe",
      "created_at": 1747852800000,
      "kind": "AssistantText",
      "data": { "text": "..." }
    },
    {
      "id": 12346,
      "session_id": "different-session",
      "agent_name": "vanto",
      "created_at": 1747852805000,
      "kind": "InjectedMessage",
      "data": { "text": "...", "from_session_id": "..." }
    }
  ],
  "next_since_id": 12346,
  "has_more": false
}
```

- `since_id` is the durable cursor — pass back what came as `next_since_id`
  on your previous call to resume cleanly.
- `since_ms` is a wall-clock-relative window (Unix milliseconds). Best-effort
  — wall-clock ties can be ambiguous; prefer `since_id` for "exact resume."
- Default (no cursor) returns the most recent `limit` events, ascending.
- `limit` defaults to 200, max 1000.
- Same `kind` translation as `/history` — including `InjectedMessage`
  surfacing for cross-session injections.

## Usage tracking

Every provider call (every iteration of the model→tools→model loop within
a user turn) emits a `turn_usage` event with token counts pulled from the
provider's response metadata. Two consumers:

- **Live UI**. The CLI prints a dim indicator after each turn:
  `[12,440/200,000 ctx (6.2%) · out 348 · claude-sonnet-4-6]`. When the
  model's context ceiling is unknown, the percentage is omitted.
- **Persistent record**. Each event is also written to the session's
  message log as a `TurnMetrics` row. These rows are filtered out before
  the message list is sent to the next provider call (they're telemetry,
  not conversation), but `GET /history` returns them so clients can sum
  token usage across a session.

**Context-ceiling source of truth.** The harness ships a small table of
known model → max input tokens in [ark/models.py](../ark/models.py).
You can override per-agent in config:

```json
"agents": {
  "scribe": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_context_tokens": 1000000   // opt into Anthropic's 1M beta
  }
}
```

If neither the table nor a config override has a value, `context_window`
is `null` in the event and the CLI shows raw counts only.

**Automatic compaction.** When a session's fill approaches the model's
window, Ark summarizes prior conversation into a `CompactionSummary`
message and folds that summary into the system prompt from that point
forward, hiding the older turns from the LLM (but keeping them in
history for audit and client rendering). See [Compaction](#compaction)
below. If compaction is disabled or its summarizer call fails, the
underlying `context_too_long` error surfaces as before and the recovery
recipe (start a new session) still applies.

### Cache telemetry

`turn_usage` events and `TurnMetrics` rows carry `cached_input_tokens`
and `cache_write_tokens`; `input_tokens` is ALWAYS the total prompt
(providers that exclude cached tokens from their raw count are
normalized). With `agents.<name>.prompt_caching` (default true), Anthropic
paths (native + `anthropic/*` via OpenRouter) get `cache_control` markers
on the system block and the conversation tail, so the repeated prefix is
served from cache.

## Error tracking

Provider exceptions are caught inside `run_user_turn`, classified into one
of four codes, persisted as a `RunError` message, and surfaced over the WS
as an `error` event:

| Code | Triggered by |
|---|---|
| `context_too_long` | "context length exceeded" / "prompt is too long" / "input is too long" from any provider |
| `rate_limit` | 429s, "rate limit" in the message, or a `RateLimit*` exception type |
| `auth` | 401s, "authentication" / "invalid api key" in the message, or `Authentication*` exception types |
| `other` | Anything else |

After an error, the run loop ends with `stop_reason: "error:<code>"`. The
session is not deleted — history is fully readable and you can attempt
another turn (which will likely hit the same problem until you act on the
code).

## Compaction

When a session grows large, Ark automatically summarizes prior turns into a
`CompactionSummary` message and folds that summary into the system prompt.
Subsequent turns see:

```
<agent persona / environment / session context / project framing>
+ "Prior conversation (summarized): <summary text>"
+ conversation turns since the compaction
```

The pre-compaction turns stay in history — `GET /history` returns them and
clients can render them as a collapsed-by-default region below the summary
divider. But the LLM sees only the summary + fresh turns. Every compaction
is one durable message row; multiple compactions accumulate as an audit
trail of what got dropped and when.

### Triggers

| Trigger | When | `reason` value |
|---|---|---|
| **Proactive** | At turn start, if last observed `TurnMetrics.input_tokens` ≥ `compaction_threshold × context_window` | `auto:threshold(<tokens>/<window>)` |
| **Reactive** | On `context_too_long` at the first iteration of a turn (retries the same turn once with the compacted history) | `reactive:context_too_long` |
| **Client-invoked (server-generated)** | `POST /agents/{name}/sessions/{sid}/compact` with empty body — server runs the summarizer | `client-invoked` |
| **Client-invoked (client-supplied)** | Same endpoint with `{"summary": "..."}` — text is used verbatim, no LLM call | `client-supplied` |

Reactive only runs at the start of a turn — mid-tool-loop failures fall
through to the normal error path (compacting across an unmatched
`ToolCall`/`ToolResult` boundary would confuse the retry).

Compaction attempts at most once per turn. If reactive compaction succeeds
but the retry itself hits `context_too_long` again, the session fails with
`error:context_too_long` (existing recovery: start a new session).

### Config

Per-agent, both optional (defaults shown):

```json
"agents": {
  "scribe": {
    ...
    "compaction_enabled": true,
    "compaction_threshold": 0.85
  }
}
```

Threshold is a fraction strictly between 0 and 1. Setting
`compaction_enabled: false` keeps the old "just fail with `context_too_long`"
behavior for the automatic triggers; when the threshold would trigger under
that setting, a `compaction_skipped` event fires so clients can warn the
user. **The manual endpoint (`POST .../compact`) runs regardless of this
flag** — the client is explicitly asking.

### Manual compaction

```
POST /agents/{name}/sessions/{sid}/compact
Body: {}                            # server generates the summary
      { "summary": "..." }          # supplied text, no LLM call
Response (200): { "ok": true, "summary": "<text>", "reason": "client-invoked" | "client-supplied" }
Response (502): { "ok": false, "code": "<classified>", "message": "..." }
                — summarizer call failed; no CompactionSummary row was written
```

Guards:

- `404` if the agent or session doesn't exist.
- `409` if the session is mid-tool-loop (any `ToolCall` without a matching
  `ToolResult`) — compacting across that boundary would leave the retry
  seeing a `ToolResult` referencing a call id it can no longer see.
- `400` if a `summary` field is provided but empty or non-string.

Fires the same lifecycle events on `/events` as automatic compactions, so
any connected WS client sees the work happening.

CLI slash command mid-chat:

```
you> /compact                             # server-generated
you> /compact set: <your summary text>    # supplied
```

### Events

Every compaction attempt emits a lifecycle event pair on `/events`:

| Event | Fields | When |
|---|---|---|
| `compaction_started` | `reason`, `input_tokens`, `context_window`, `model` | About to run the summarizer |
| `compaction_completed` | `summary`, `reason` | Summary persisted; subsequent turns will use it |
| `compaction_failed` | `code`, `message`, `reason` | Summarizer call errored — the turn either continues uncompacted (proactive) or falls to `context_too_long` (reactive) |
| `compaction_skipped` | `reason`, `input_tokens`, `context_window` | Threshold crossed but `compaction_enabled: false` — a warning signal, no action taken |

Clients can render "Compacting session… (context was 87% full)" between
`compaction_started` and `compaction_completed`, and show the resulting
summary alongside the visual divider in the transcript.

### What the summarizer preserves

The default summarizer prompt asks the model (the session's own model —
same provider, same context) to preserve names, facts, decisions, files
referenced by path, code discussed or written, commitments made to the
user, open questions, and significant tool results. It's told to omit
persona/environment (those are provided separately) and to be complete
over concise.

If a prior `CompactionSummary` exists, its text is passed to the new
summarizer as background so information isn't lost across successive
compactions.

### Sharp edges

- **The summarizer's judgment is load-bearing.** A bad summary drops
  something the user cared about. Older summaries stay in history as an
  audit trail; a future "restore" flow could rewind past them.
- **Cost.** One extra provider call per compaction, using the session's
  own model. Later versions may allow a cheaper `compaction_model`
  override.
- **Guard floors.** Compaction skips when fewer than 6 messages have
  accumulated since the last summary — no point summarizing a handful of
  turns.
- **First turn.** With no `TurnMetrics` observed yet, the proactive check
  has no denominator; it skips.
- **Post-compaction fill unknown.** Immediately after a compaction, the
  last `TurnMetrics` still shows the pre-compaction count. The proactive
  check detects this and skips until the next turn's fresh metrics land.

## Cross-session messaging

An agent can inject a message into another of its own sessions via the
built-in `post_to_session` tool. The receiving session records the message
in history and pushes a `file_available`-style `injected_message` event to
any connected WS clients. See [design/design.md §8](../design/design.md) for
the rationale; the implementation lives in [ark/broker.py](../ark/broker.py)
and the `post_to_session` tool in [ark/tools.py](../ark/tools.py).

## Session metadata + cron fire history

For debugging "what did the cron actually do," two endpoints + two CLI
commands:

```
GET /sessions/{sid}                                 # session metadata
GET /agents/{name}/crons/{cron_id}/sessions[?limit] # fires of a specific cron
```

The metadata endpoint returns `{id, agent_name, kind, created_at, ended_at,
project_id, cron_id, cron_prompt?}`. `cron_prompt` is present only when the
session is a cron fire — it's the prompt from the cron entry at the time the
fire was rendered, which makes transcripts self-explanatory.

The fire-history endpoint returns each fire enriched with a one-line
`summary` (the first `post_to_session` body, falling back to last
`AssistantText`, falling back to `"(no output)"`), plus `had_error` and
`error_code`. Clients can render a table without round-tripping `/history`
per row.

```bash
ark cron history <agent> <cron-id> [--limit N]
ark show <session-id>
```

`ark show` pretty-prints any session — cron, heartbeat, or conversational —
collapsing turns and surfacing `RunError` rows + token-usage metrics inline.

`sessions.cron_id` is populated only for sessions created by the scheduler
firing that cron. Historical sessions (pre-migration) keep a null
`cron_id` and won't surface in the new history endpoint.

## Heartbeat and cron sessions

When a heartbeat fires or a cron expression matches, the scheduler creates
a fresh session of kind `heartbeat` or `cron` and runs the same turn loop a
conversational session uses. The starting prompt comes from
`<ARK_HOME>/agents/<name>/heartbeat_prompt.md` (heartbeats) or the cron
entry's `prompt` column (crons). Scheduled sessions can post to
conversational sessions via `post_to_session` to surface results to humans.

Adding per-session context to a scheduled session is unusual but works —
the same REST endpoint accepts any session id regardless of kind.

**Cron entries can also carry a `project_id`**, in which case each fire
creates a session already bound to that project (system prompt gets the
project stanza from turn 1, uploads land in the project's dir). See
[projects.md § Cron entries can be bound to a project](projects.md#cron-entries-can-be-bound-to-a-project).
