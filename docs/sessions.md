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
Body (optional): {
  "context"?: "...",
  "project_id"?: "<uuid>",
  "metadata"?: { ... }               // opaque server-only JSON object
}
→ { "id": "<uuid>" }
```

Always creates a `conversational` session. If `context` is provided, it's
appended as the session's first `SessionContext` message — see
[Per-session context](#per-session-context) below. If `project_id` is
provided, the session is bound to that project — see
[projects.md § Binding a session to a project](projects.md#binding-a-session-to-a-project).
If `metadata` is provided, it's stored server-side and made available to
skills — see [Session metadata](#session-metadata) below.

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
`CompactionSummary`, `ProjectAssignmentChanged`, `DateMarker`).

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

## Session metadata

A parallel channel to per-session context, but **server-only** — never
rendered into the system prompt, never in the LLM message list, never
surfaced by any read API. Set once at session creation via the
`metadata` field on `POST /agents/{name}/sessions`; available to skills
via `current_context().metadata`.

```
POST /agents/{name}/sessions
Body: { "metadata": { "callback_url": "...", "callback_secret": "..." } }
```

The value must be a JSON object (`400` otherwise). Skills read it as:

```python
from ark.tools import current_context

def my_tool():
    ctx = current_context()
    callback_url = ctx.metadata.get("callback_url")   # {} if none supplied
    ...
```

### Why it exists

`SessionContext` text ends up in the model-visible conversation, where a
prompt-injection attack — a hostile document, a compromised web page —
can trick the agent into leaking or misusing the text. That's fine for
persona/behavior nudges but disastrous for capability credentials.

Metadata is deliberately kept on the same unforgeable server-side channel
as `session_id`: the model can neither observe nor mutate it. Use it for
per-session capabilities the skill needs but the model must never see:

- Callback URL + secret pair for a client-side tool gateway
- Per-tenant API tokens the skill will pass to an upstream service
- User identity/permissions the skill will check before acting

### Guarantees

- **Not in the system prompt.** The prompt is built from persona,
  environment, project, and `SessionContext` — never from metadata.
- **Not in the message list.** Metadata never appears as a `UserText` or
  any other message kind sent to the provider.
- **Not surfaced by read APIs.** `GET /sessions`, `GET /sessions/{sid}`,
  `GET /agents/{name}/sessions/{sid}/history` all omit metadata. It
  exists only on the row and in `ToolContext.metadata` during a turn.
- **Immutable within a session.** Set at creation; no update endpoint in
  v1. Rotating a callback secret today means starting a new session.
  (`PATCH /sessions/{sid}/metadata` is a natural follow-on if needed.)
- **Read-only from skills.** `ctx.metadata` is a dict, but writes from a
  skill don't persist — this is by design: writable metadata would let
  the model indirectly influence the channel, defeating the purpose.

### Empty by default

`ctx.metadata` is `{}` when nothing was supplied. Skills that expect
specific keys should handle absence explicitly rather than relying on
the caller.

## Date markers

Long-running sessions tend to confuse the model about time. If a user
sends a message, then comes back 2 days later and sends another, the
model's sense of "today" and "yesterday" is still anchored to the
conversation's earlier turns, not the current wall-clock date.

Ark addresses this at two levels:

### Always-fresh date in the system prompt

The Environment stanza rebuilds per turn with the current date. When
the client includes `timezone` on their `user_message` frame, that
zone's local date is shown with UTC in parentheses:

```
- Today's date (America/Los_Angeles): 2026-10-05  (UTC: 2026-10-06).
  Your training data has a cutoff; treat this field as the ground
  truth for "today" and defer to `get_current_time` (returns UTC) when
  you need the exact wall-clock time.
```

With no client timezone (UTC default), the line is just
`Today's date (UTC): 2026-10-05`.

Correctness-level guarantee: the model always has the right date
visible, independent of history age.

### `DateMarker` injection

When a user turn starts and the calendar date (in the client's
timezone) differs from the previous `UserText`'s date (same zone), the
runtime inserts a `DateMarker` row into history **before** persisting
the new user message. The model sees a synthetic user turn via
`_rewrite_for_llm`:

> `[system notification: Time has passed since the last turn in this session. The current date is 2026-10-05 (America/Los_Angeles). The previous turn was on 2026-09-29 (6 days ago). Your prior sense of "today", "yesterday", or recent events may be stale — recalibrate accordingly.]`

The explicit "recalibrate" line is the behavioral nudge. A static date
line in the system prompt tends to be read as background; a synthetic
turn mid-conversation is louder.

### Timezone semantics

The timezone used for the date comparison is **client-supplied per
turn** via the `timezone` field on the `user_message` WS frame (IANA
zone name, e.g. `"America/Los_Angeles"`). The CLI auto-detects the
system's local zone and sends it on every turn.

Rationale: the timezone is a property of the client at the moment of
the turn, not of the agent server-side. A user on their phone in NYC
and their laptop in SF each get correct markers for their current
location without any server config.

- **Both ends of the comparison** use the current turn's zone — "from
  the client's current frame of reference, has the date changed?"
  Simple semantic that handles travel-between-turns cleanly.
- **Fallback**: absent, invalid, or non-string `timezone` → UTC
  (unchanged pre-feature behavior).
- **Persistence**: the `DateMarker` row records the TZ it was computed
  in, so `/history` and the `date_marker` wire frame expose it.
  Clients rendering divider UI can show "── Oct 5 (LA time) ──".

### Scope

- **Conversational sessions only.** Cron/heartbeat sessions have their
  own temporal framing baked into their session kind (they're always
  "right now").
- **Calendar-date granularity.** No elapsed-hours trigger — a short gap
  within the same local day doesn't fire a marker. Date-change is
  sharp, deterministic, and the right unit for the "I came back later"
  case.
- **Previous `UserText` row** is the comparison anchor, not the last
  message of any kind. Cron-fired `post_to_session` activity between
  user turns doesn't hide the "user came back" signal.

### What clients see

A real `DateMarker` row in `/history`:

```json
{
  "kind": "DateMarker",
  "data": {
    "from_date": "2026-09-29",
    "to_date": "2026-10-05",
    "elapsed_days": 6,
    "timezone": "America/Los_Angeles"
  }
}
```

Live `date_marker` WS frame on `/events`:

```json
{
  "type": "date_marker",
  "session_id": "...", "agent_name": "scribe",
  "from_date": "2026-09-29",
  "to_date": "2026-10-05",
  "elapsed_days": 6,
  "timezone": "America/Los_Angeles",
  "event_id": 2201
}
```

Rendering is the client's call. Common policies:

- **Hide entirely** — filter `kind === "DateMarker"` out of the chat view.
  Matches how clients typically handle other system markers.
- **Render as a subtle divider** — `── Oct 5, 2026 (6 days later) ──`
  between messages, using the structured fields (not the model-facing
  notification text, which never reaches the client).
- **Loud banner** — unusual, but possible for an "it's been a while"
  callout.

The model-facing "[system notification: …]" string exists only in the
message list sent to the provider. It's not persisted and not returned
by any read API.

### Sharp edges to know

1. **Backfill**: existing sessions have no markers in history. The first
   new turn in an old session may fire a marker with an elapsed_days of
   weeks or months. That's actually desirable — it tells the model about
   the gap.
2. **Compaction interaction**: the marker is a durable row. If
   compaction runs across one, the summarizer sees the synthetic
   notification in its input and should preserve the "last known date"
   context (default summarizer prompt preserves significant events).
3. **No elapsed-time trigger**. If a session stays idle for 20 hours
   within the same UTC date (common for US-timezone users spanning an
   evening break), no marker fires. If this becomes a problem in
   practice, an elapsed-time trigger can be added later as a separate
   option.
4. **Timezone**: the marker's `from_date`/`to_date` are in whatever
   zone the client supplied on the `user_message` frame (`timezone`
   field). Clients that render the divider can either use the stored
   date strings as-is or re-convert from the matching `UserText` row's
   `created_at` + their own current zone.

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
| `date_marker` | `from_date`, `to_date`, `elapsed_days` | A UTC-date boundary was crossed since the previous user turn — the "time has passed" cue (see [Date markers](#date-markers) below) |
| `done` | `stop_reason`, `stopped?` | Whole run-loop finished for that session, awaiting next user input. On classified errors, `stop_reason` is `"error:<code>"`. On a `stop`-triggered cancel, `stop_reason` is `"stopped"` and the event carries `stopped: true`. |

Every event also carries `session_id` and (except for the broad "error" case
where the session couldn't be identified) `agent_name`.

### Event ids

Events that correspond to a **persisted `messages` row** carry an
`event_id` field on the wire — the same integer `messages.id` that
`GET /events` returns for the row in catch-up.

Purpose: lets clients maintain a **durable cursor** that advances from
the live WS AND catch-up REST uniformly. On reconnect, a client fetches
`GET /events?since_id=<last_seen_event_id>` and skips whatever it already
saw via the live stream — no time-window text-hash dedupe required.

| Wire event | Persisted? | `event_id` |
|---|---|---|
| `assistant_delta` | No (streamed segments — final `AssistantText` at turn end) | No |
| `assistant_message` | Yes (`AssistantText`) | **Yes** (when the turn produced text) |
| `thinking` | No | No |
| `tool_call` | Yes (`ToolCall`) but persisted at TurnEnd, after this frame ships | **No** — see note below |
| `tool_result` | Yes (`ToolResult`) | **Yes** |
| `turn_usage` | Yes (`TurnMetrics`) | **Yes** |
| `error` | Yes (`RunError` — includes the outer-catch escape path) | **Yes** |
| `compaction_started` / `_failed` / `_skipped` | No (lifecycle markers) | No |
| `compaction_completed` | Yes (`CompactionSummary`) | **Yes** |
| `session_project_changed` | Yes (`ProjectAssignmentChanged`) | **Yes** |
| `date_marker` | Yes (`DateMarker`) | **Yes** |
| `injected_message` | Yes (`AssistantText` in target session) | **Yes** |
| `file_available` | Yes (`SharedFile`) | **Yes** |
| `workspace_file_changed` / `project_file_changed` | No (external FS events) | No |
| `done` | No (per-run terminator) | No |

**Client contract**: `event_id` absent → don't advance cursor. `event_id`
present → cursor := max(cursor, event_id). On reconnect,
`GET /events?since_id=cursor` picks up everything the live socket
missed, with no duplication of what the live socket already delivered.

**About `tool_call`**: the `ToolCall` row is persisted at the end of the
assistant turn (with `AssistantText`), not at the moment the streaming
frame is yielded. Adding `event_id` there would require reordering the
DB `seq` in a way that changes what the provider sees on subsequent
turns (Anthropic in particular expects assistant text before tool_use in
each block). Clients that want to reference a specific tool call by
persisted-row id can consume the corresponding `tool_result` frame's
`event_id` and read the paired `ToolCall` row from `/history`, or rely
on the tool-call correlation id in the frame's `id` field for live-only
matching.

**About the outer-catch `error` path**: if something escapes
`run_user_turn` entirely (a programming error, a broker failure, etc.),
the runtime now persists a `RunError` row for it too — same shape as
in-turn errors — so those `error` frames carry `event_id`. If the DB
write itself fails (unlikely but possible), the frame goes out without
`event_id` as a best-effort fallback.

**About wire naming**: the field is `event_id` (not `id`) to avoid
colliding with `tool_call.id` / `tool_result.id`, which are the
tool-call correlation ids for pairing request→response frames — a
different identifier space.

### Client → server commands

| Command | Required fields | Effect |
|---|---|---|
| `user_message` | `session_id`, `text`, optional `timezone` (IANA zone, e.g. `"America/Los_Angeles"`) | Start a new turn in that session. Multiple sessions can have turns running concurrently — events stream back tagged with their `session_id`. If `timezone` is included, the server uses it for the [Date markers](#date-markers) comparison and the Environment stanza's "today's date" line (otherwise UTC). Invalid or non-string → silent fallback to UTC. |
| `stop` | `session_id` | Cancel the in-flight turn for that session. Fire-and-forget: the cancellation lands as a terminal `done {"stopped": true, "stop_reason": "stopped"}` on the events stream. Silent no-op when no turn is running. Also terminates any in-flight `run_command` process group (SIGTERM immediately, SIGKILL after a 5s grace) so long-running shell commands don't outlive the cancel. |

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

## Error tracking

Errors are caught inside `run_user_turn` — either classified provider
exceptions or a budget breach raised by the runtime itself — persisted as
a `RunError` message, and surfaced over the WS as an `error` event:

| Code | Triggered by |
|---|---|
| `context_too_long` | "context length exceeded" / "prompt is too long" / "input is too long" from any provider |
| `rate_limit` | 429s, "rate limit" in the message, or a `RateLimit*` exception type |
| `auth` | 401s, "authentication" / "invalid api key" in the message, or `Authentication*` exception types |
| `token_budget_exceeded` | Cumulative input+output tokens across this turn's iterations exceeded the effective budget. See [Turn token budget](#turn-token-budget) below. |
| `other` | Anything else |

After an error, the run loop ends with `stop_reason: "error:<code>"`. The
session is not deleted — history is fully readable and you can attempt
another turn (which will likely hit the same problem until you act on the
code).

**Error message enrichment.** For every classified code, the `message`
field carries the exception's class name and — when the SDK exposes them
— a status code and request id. So a bare provider 5xx that would
otherwise surface as just `"Internal Server Error"` becomes something
like `"InternalServerError [status=500]: Internal Server Error
(request_id=req_abc123)"`. This is done via duck-typed attribute lookup
(`status_code`, `request_id`, `response.headers['x-request-id']`), no
SDK imports — works uniformly across Anthropic, OpenAI, OpenRouter, and
Google. Full tracebacks are also logged to stderr on every classified
error so `docker compose logs ark | grep <session-id>` gives operators
the whole stack even when the wire message is thin.

## Turn token budget

Every turn has a cumulative token budget — input+output summed across the
iterations of the model→tools loop. When the budget is exceeded (checked
between iterations, so at least one iteration always runs), the turn
terminates with `error:token_budget_exceeded`. This replaces the old
hardcoded 16-iteration cap; a runaway that produces small output per
iteration will now spin much longer before the budget catches it, and a
legitimate turn doing 30 small tool calls no longer dies at iteration 16.

**Precedence** (highest wins):

1. Explicit `max_tokens` arg to `run_user_turn` / `run_and_publish` — the
   scheduler passes `crons.max_tokens` here.
2. `AgentConfig.max_turn_tokens` — per-agent override in config.
3. `runtime.DEFAULT_TURN_TOKEN_BUDGET` — currently `500_000`. Generous by
   design — ordinary turns run 5–25k total.

**Metric**: cumulative `input_tokens + output_tokens` from every
`TurnMetrics` row written during this turn. Compaction's summarizer call
does NOT emit `TurnMetrics` and is not counted toward the budget.

**Config**:

```json
"agents": {
  "scribe": {
    ...
    "max_turn_tokens": 2000000
  }
}
```

Positive integer or omit for the default. Per-cron override lives on the
cron row itself — see [projects.md § Cron entries can be bound to a project](projects.md#cron-entries-can-be-bound-to-a-project).

## Per-response output cap

Separate from the turn budget above: `max_output_tokens` on `AgentConfig`
sets the per-response cap the runtime passes to `provider.stream_turn(...)`
on every call — the SDK's `max_tokens` kwarg. Falls back to **4096** when
unset (matching every provider adapter's def-time default).

```json
"agents": {
  "scribe": {
    ...
    "max_output_tokens": 16000
  }
}
```

**What it caps**: how much a single response from the model can be. A
turn that runs 10 iterations of the model→tools loop can produce
`10 × max_output_tokens` of output; this only bounds each iteration
individually.

**When to raise it**: agents that legitimately produce long single
responses — drafting a document, writing a large code block, producing a
detailed plan. The 4096 default truncates all of these mid-sentence with
no client-visible warning beyond `stop_reason: "max_tokens"` on `done`.

**When you can't set it to zero**: Anthropic requires `max_tokens` on
every API call, so there is no "uncapped" option. OpenAI and Gemini
would allow it, but Ark doesn't expose that variant today.

**Truncation behavior**: when the model hits the cap mid-generation, the
partial output is persisted, the run loop terminates naturally, and the
client sees `done` with `stop_reason: "max_tokens"`. No `RunError`.
Clients that want to make truncation visible can special-case that
`stop_reason` in their UI.

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
