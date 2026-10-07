# Changelog

## Unreleased — Date markers in long-running sessions

Long-running conversational sessions tend to confuse the model about
time: a user sends a message, comes back two days later, sends another,
and the model's sense of "today" is still anchored to the conversation's
older turns. This feature addresses that at two levels. See
[docs/sessions.md § Date markers](docs/sessions.md#date-markers).

### Always-fresh date in the system prompt

The Environment stanza rebuilds per turn with the current date. With a
client-supplied timezone (see below) it shows the local date + UTC in
parens:

```
- Today's date (America/Los_Angeles): 2026-10-05  (UTC: 2026-10-06).
  Your training data has a cutoff; treat this field as the ground
  truth for "today"...
```

Without a timezone, it falls back to `Today's date (UTC): …`.

### `DateMarker` injection

When a user turn starts and the calendar date (in the client's
timezone) differs from the previous `UserText`'s date, the runtime
persists a `DateMarker` row before the new user message. The model
sees a synthetic UserText notification via `_rewrite_for_llm`:

> `[system notification: Time has passed since the last turn... The current date is 2026-10-05 (America/Los_Angeles). The previous turn was on 2026-09-29 (6 days ago). ... recalibrate accordingly.]`

The explicit "recalibrate" nudge is the behavioral lever a static
system-prompt line can't provide.

### Client-supplied timezone (per turn)

The `user_message` WS frame now accepts an optional `timezone` field
(IANA zone name — `"America/Los_Angeles"`, `"Europe/London"`, etc.).
Used for the date-change comparison + env-stanza "today's date" line.

```json
{
  "type": "user_message",
  "session_id": "...",
  "text": "...",
  "timezone": "America/Los_Angeles"
}
```

- Rationale: timezone is a property of the client at the moment of the
  turn, not of the agent. Travel + multi-device + future multi-user all
  work correctly without server config.
- **Both ends of the date comparison** use the current turn's zone —
  "from the client's current frame of reference, has the date changed?"
- **Fallback**: absent / invalid / non-string → UTC (preserves pre-TZ
  behavior).
- **Persistence**: the `DateMarker` row records which TZ it was
  computed in. `/history` and the `date_marker` wire frame both expose
  it.
- **CLI**: `ark chat` auto-detects the system's IANA zone (via
  `/etc/localtime` symlink, falling back to `$TZ`) and sends it on
  every turn — users get correct date transitions for free.

### Scope limits (v1)

- Conversational sessions only — cron/heartbeat have their own temporal
  framing.
- Calendar-date change only — no elapsed-hour trigger (idle but
  same-local-day doesn't fire).
- Previous `UserText` row is the comparison anchor — cron-fired
  `post_to_session` activity between user turns doesn't hide the "user
  came back" signal.

### New wire shape

Live `date_marker` event on `/events`:

```json
{
  "type": "date_marker",
  "session_id": "...", "agent_name": "scribe",
  "from_date": "2026-09-29", "to_date": "2026-10-05",
  "elapsed_days": 6,
  "timezone": "America/Los_Angeles",
  "event_id": 2201
}
```

`event_id` matches the `DateMarker` row's `messages.id` — durable cursor
advances uniformly across live + catch-up.

In `/history`, the row appears as `kind: "DateMarker"` with structured
fields (including `timezone`). Rendering is the client's call: hide,
subtle divider ("── Oct 5, 2026 (6 days later, LA time) ──"), loud
banner. The model-facing "[system notification: …]" text never reaches
the client — only the structured form.

### New types + API additions

- `DateMarker(from_date, to_date, elapsed_days)` message type +
  round-trip in `message_to_row` / `message_from_row`.
- `DateMarkerEvent(from_date, to_date, elapsed_days, row_id)` runtime
  event with wire conversion.
- `runtime._maybe_insert_date_marker(conn, session_id)` — fires at
  `run_user_turn` entry.
- `runtime._date_marker_notification(msg)` — renders the LLM-facing
  text.
- `_rewrite_for_llm` substitutes `DateMarker` → synthetic UserText.
- `system_prompt` env stanza always carries today's UTC date.

### Sharp edges

- **Backfill**: existing sessions have no markers. The first turn in an
  old session may fire a marker with a large `elapsed_days` — desired,
  tells the model about the gap.
- **Compaction interaction**: default summarizer preserves significant
  events; the marker should carry forward naturally.
- **No elapsed-hour trigger**. Same-UTC-day idle periods don't fire.
  Add as a follow-on if needed.
- **UTC** means the "date change" happens at a different wall-clock
  time depending on local TZ. Clients rendering local-date dividers may
  want to transform `to_date`.

### Tests

`tests/test_date_marker.py` (14): round-trip, env-stanza date presence,
insertion fires on date change, skips same-day / first turn /
cron+heartbeat kinds / non-UserText anchor, `_rewrite_for_llm`
substitution + synthetic-user-text text, wire format with/without
`event_id`, end-to-end runtime yields `DateMarkerEvent` + LLM sees
notification + system prompt has today's date, baseline back-to-back
same-day case fires no marker.

## Unreleased — `event_id` on live WS events for durable-cursor dedupe

Live `/events` WS frames that correspond to a persisted `messages` row
now carry `event_id` — the same integer `messages.id` that `GET /events`
returns on catch-up. Clients can maintain a durable cursor across the
live and catch-up surfaces uniformly; on reconnect,
`GET /events?since_id=<cursor>` picks up the gap without re-delivering
anything the live socket already emitted. See
[docs/sessions.md § Event ids](docs/sessions.md#event-ids).

### Motivation

Previously, live WS events had no id. A client staying up for hours
between catch-ups would never advance its cursor, so the next
`GET /events` re-fetched everything back to the last catch-up — every
one of which the client already processed live. The workaround was
substring-containment SQL dedupe in a 30-day window, expensive and
wrong for legitimately-repeated cron output.

Multi-segment turns compounded the pain: the live path emits many
`assistant_delta` + one `assistant_message`; catch-up sees one
`AssistantText` row. Without a stable per-event identifier, no dedupe
strategy worked cleanly across both shapes.

### What lands on the wire

Every event whose underlying row is persisted gains `event_id`:

- `assistant_message`, `tool_result`, `turn_usage`, `error`,
  `compaction_completed`, `session_project_changed`, `injected_message`,
  `file_available`

Ephemeral events (streaming deltas, lifecycle-only compaction frames,
external filesystem events, per-turn `done`) carry no `event_id`. Client
contract: **no `event_id` → don't advance cursor**. Persisted-row events
are the exclusive source of cursor advancement.

### One deliberate exception

`tool_call` (live) does NOT carry `event_id`. The `ToolCall` row is
persisted at TurnEnd — after this frame ships — and reordering the DB
`seq` to persist earlier would change what provider adapters see on
subsequent turns (Anthropic in particular expects text-before-tool_use
in assistant blocks). Clients dedupe against the paired `tool_result`
frame (which has `event_id`) or consume `/history` for the durable
identity.

### One new behavior worth flagging

**The outer-catch `error` path now persists a `RunError` row.** Prior
behavior: unhandled escapes from `run_user_turn` (programming errors,
broker failures, etc.) published an `error` frame but wrote nothing to
history. Now they persist a `RunError` too, so the frame carries
`event_id` and the error shows up in `/history` and catch-up. Closes
the "no `event_id` → cursor stalls" gap for the rare-but-real class of
unhandled escapes. If the DB write itself fails (very unlikely), the
frame still goes out — just without `event_id`, as a best-effort
fallback.

### Wire naming: `event_id`, not `id`

`tool_call.id` and `tool_result.id` already meant the tool-call
correlation id (for pairing request→response frames). Using `id` for
the messages row would collide. `event_id` is the additive field name
used everywhere.

### API changes

- **`runtime.append_message(conn, session_id, msg) -> int`** — now
  returns the new row's globally-monotonic `messages.id`. Callers that
  ignored the return before continue to work.
- **`runtime.set_session_project(...)`** — now returns a 3-tuple
  `(from_project, to_project, marker_row_id)` on a real change (was
  2-tuple); no-op still returns `None`.
- **`RuntimeEvent` dataclasses** — `AssistantTurnEnd`, `ToolCallEvent`,
  `ToolResultEvent`, `TurnUsageEvent`, `RunErrorEvent`,
  `CompactionCompletedEvent` gain optional `row_id: int | None = None`.
  All defaults are `None`, so every existing construction site is
  unaffected.
- **`event_to_wire`** — persists `event_id` on the wire when the event
  carries a `row_id`.

### Tests

`tests/test_event_ids.py` (16 tests): `append_message` return value,
wire-format `event_id` presence/absence per event kind, end-to-end
runtime tests that live event ids match persisted row ids, broker
publish sites (session_project_changed, client-supplied compaction) fire
with `event_id`, outer-catch escape persists + carries id, and a
catch-up alignment test proving live `event_id == GET /events id` for
the same row.

## Unreleased — Per-agent per-response output cap (`max_output_tokens`)

Every provider adapter's `stream_turn` accepts `max_tokens` and defaults
to `4096`; `run_user_turn` never overrode it, so 4096 was effectively
hardcoded. Agents that produced long single responses (drafts, plans,
large code blocks) got truncated mid-sentence with no way to raise the
cap in config.

Adds a per-agent `max_output_tokens` field on `AgentConfig`. When set,
the runtime passes it as `max_tokens` to `stream_turn` on every call.
When absent (the default), behavior is unchanged from before this PR.

```json
"agents": {
  "scribe": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_output_tokens": 16000
  }
}
```

Positive integer. Absent → 4096. See
[docs/sessions.md § Per-response output cap](docs/sessions.md#per-response-output-cap).

### Distinct from the other three token controls

- **`max_output_tokens`** (this): caps a single provider response.
- **`max_turn_tokens`** (already shipped): caps cumulative input+output
  across all iterations of one turn.
- **cron `max_tokens`** (already shipped): per-cron override for
  `max_turn_tokens`.
- **Compaction/context window** (already shipped): caps input via
  auto-summarization.

They stack. This one is the one that fixes "the model's response cut
off at 4096 tokens mid-sentence."

### From PR #1

Landed via [#1](https://github.com/druths/ark/pull/1) rebased onto
current main and renamed `max_tokens` → `max_output_tokens` before
merge. The original name would have collided with `crons.max_tokens`
(per-turn budget) — same word, different scope. The new name is
unambiguous and matches Google's SDK naming exactly.

## Unreleased — Opaque server-only session metadata

Adds a `metadata` field to session creation — an opaque JSON object stored
server-side and surfaced to skills via `ToolContext.metadata`, but
**never** rendered into the system prompt, the LLM message list, or any
read API. See [docs/sessions.md § Session metadata](docs/sessions.md#session-metadata).

### Threat model this addresses

`SessionContext` text is model-visible, so a prompt-injection attack (a
hostile document, a compromised web page) can trick the agent into
leaking or misusing whatever it contains. That's fine for
persona/behavior nudges but disastrous for capability credentials
(callback URLs + secrets, per-tenant API tokens, upstream permissions).

Metadata is deliberately kept on the same unforgeable server-side
channel as `session_id`: the model can neither observe nor mutate it.

### Schema migration v7

`ALTER TABLE sessions ADD COLUMN metadata_json TEXT` (nullable). Existing
sessions have `NULL`; skills reading `ctx.metadata` get `{}`.

### API changes

- **`POST /agents/{name}/sessions`** accepts optional `metadata: {...}`
  in the body. `400` if present but not a JSON object.
- **`runtime.create_session(..., metadata=None)`** — new kwarg. Existing
  callers unaffected.
- **`runtime.session_metadata(conn, sid)`** — read-back helper returning
  `{}` when absent, malformed, or not an object.
- **`ToolContext.metadata`** — new field, default `None`. Populated to
  the session's metadata during a turn. Every existing `ToolContext(...)`
  construction site keeps working (the default preserves the signature).

### Guarantees (tested)

- Not in the system prompt.
- Not in the message list sent to the provider (leak test in the suite
  asserts the value appears in neither `system` nor `messages` during a
  tool-calling turn).
- Not surfaced by `GET /sessions`, `GET /sessions/{sid}`, or the history
  endpoint.
- Read-only from skills — the field is a dict, but there's no persistence
  path from `ctx.metadata` writes (by design).

### Sharp edges

- **Immutable within a session.** Set at creation only; no PATCH endpoint
  in v1. Rotating a callback secret means starting a new session.
- **Empty by default.** Skills should handle missing keys defensively.
- **Server-only.** Restoring metadata from a backup requires the DB row;
  it isn't reconstructable from message history.

### From PR #3

Landed via [#3](https://github.com/druths/ark/pull/3) rebased onto
current main. Two adjustments during rebase:
- The PR's stop-cancel commit is already in main (via #6) — cherry-picked
  only the metadata commit.
- Migration renumbered from 5 → 7 (main already has 5 for
  `crons.project_id` and 6 for `crons.max_tokens`).

## Unreleased — Real mid-turn cancellation for the `stop` command

The `stop` WS command was previously a documented no-op — turn tasks
were spawned via `asyncio.create_task(...)` and immediately discarded,
so there was nothing to cancel with. Now it actually cancels the turn
and terminates any in-flight shell commands.

### How

- **Turn registry.** `run_and_publish` registers `asyncio.current_task()`
  in a per-session dict on entry and cleans up in `finally`. Every spawn
  site — the WS handler, scheduler heartbeats and crons — gets cancel
  support without call-site changes.
- **`runtime.stop_turn(session_id)`** cancels the registered task if
  one's running; returns `False` when nothing's in flight.
- **Terminal event on cancel.** The task publishes
  `done {"stopped": true, "stop_reason": "stopped"}` from a
  `CancelledError` branch in `run_and_publish` before re-raising, so
  clients see a clean turn end instead of a mid-stream drop.
- **Process-group SIGTERM → SIGKILL discipline.** `run_command` now
  runs in its own process group (`start_new_session=True`) and registers
  each `Popen` per session. `tools.stop_session_commands(session_id)`
  SIGTERMs the group immediately and escalates to SIGKILL after a 5s
  grace via a daemon timer thread. Task cancellation alone can't reach
  the subprocess: the command runs under `asyncio.to_thread`, and
  cancelling the awaiter leaves the worker thread and its subprocess
  running to timeout. **Side benefit**: the existing tool-timeout path
  now also kills children, not just the shell.
- **`stop` handler in the WS loop.** Cancels the turn + kills the
  session's in-flight commands; fire-and-forget shape like
  `user_message`. Silent no-op when nothing's running; an `error`
  frame only when `session_id` is missing or non-string.

### Test-injection fix rolled in

`run_user_turn`'s `provider_factory=make_provider` default was bound
at definition time — so the `runtime.make_provider` monkeypatch
pattern that several tests already relied on never actually injected
through `run_and_publish`. The default is late-bound now (`None →
resolve to `make_provider` at call time`), matching what
`compact_session` was already doing.

### New tests

`tests/test_stop_cancel.py` — 6 tests: WS-level mid-stream cancel
ending in `done {stopped:true}` with the session immediately usable on
the same socket, silent no-op when nothing's running,
missing-`session_id` error frame, registry cancel + hygiene, and a
process-group kill test (`sleep 30` terminated with exit code `-15`).

### From PR #2

Landed via [#2](https://github.com/druths/ark/pull/2) rebased onto the
current main. The one conflict was `run_user_turn`'s signature —
resolved by keeping the token budget (which supersedes the old
`max_iterations` param the PR touched) and taking the PR's late-bound
`provider_factory` fix.

## Unreleased — Richer error messages

Provider errors surfaced on `/events` and persisted as `RunError` rows
now include the exception class name and (when the SDK exposes them) the
HTTP status code and request id. A bare `"Internal Server Error"` from a
provider 5xx now becomes `"InternalServerError [status=500]: Internal
Server Error (request_id=req_abc123)"` — enough context to open a
support ticket without digging in the logs.

Extraction is duck-typed (`status_code`, `request_id`,
`response.headers['x-request-id']` / `request-id` /
`x-anthropic-request-id`) so the runtime doesn't couple to any
provider-SDK version.

Full tracebacks are also logged to stderr on every classified error and
on any unhandled exception that escapes `run_user_turn` — grep
`docker compose logs ark` for the session id to find the stack.

Applies uniformly to:
- Turn errors in `run_user_turn` (persisted as `RunError`, emitted as
  `error` event).
- Compaction failures in `compact_session` (emitted as
  `compaction_failed` event).
- Unhandled escapes in `run_and_publish` (bare `error` event on the
  broker).

No wire-format change — the `message` field just carries more useful
text. Existing clients see the same shape.

## Unreleased — Turn token budget replaces max_iterations

The hardcoded 16-iteration cap on `run_user_turn`'s model→tools loop is
gone. Turns now terminate on a **cumulative token budget** — input+output
summed across iterations of the current turn. This unblocks legitimate
long-running work (e.g. a cron doing many small tool calls) while still
catching genuine runaway. See
[docs/sessions.md § Turn token budget](docs/sessions.md#turn-token-budget).

### Precedence

Highest wins:

1. Explicit `max_tokens` arg to `run_user_turn` / `run_and_publish` —
   the scheduler passes `crons.max_tokens` here.
2. `AgentConfig.max_turn_tokens` — per-agent override in config.
3. `runtime.DEFAULT_TURN_TOKEN_BUDGET` — currently `500_000`. Generous
   by design; ordinary turns run 5–25k total.

### Schema migration v6

Adds `crons.max_tokens INTEGER` (nullable). Existing crons keep firing
at whatever the agent/global default is unchanged.

### Config

Per-agent, optional, positive integer:

```json
"agents": {
  "scribe": { ..., "max_turn_tokens": 2000000 }
}
```

Existing configs pick up the default without change.

### REST + tool + CLI additions

- `PUT /agents/{name}/crons/{cron_id}` now accepts optional
  `max_tokens: int | null`. Same "omit = preserve, null = clear"
  semantics as `project_id`. `GET .../crons` returns it.
- `add_cron(id, expr, prompt, project_id?, max_tokens?)` — new optional
  parameter, validated at add time.
- `list_crons` output shows `[max_tokens=N]` next to each cron that has
  an override.
- `ark cron set --max-tokens N` / `--no-max-tokens` — mutually exclusive
  with each other; omitting both preserves the existing setting.

### New error code

`token_budget_exceeded` joins `context_too_long`, `rate_limit`, `auth`,
`other`. Persisted as `RunError`, emitted as `error` event, terminates
the turn with `stop_reason: "error:token_budget_exceeded"`. The message
includes both the actual cumulative count and the effective budget so
operators can decide whether to bump the cap or fix the workflow.

### Sharp edges

- **Check is post-iteration.** The budget is verified between iterations,
  so a single iteration that produces a huge response is recorded and
  counted, then the next iteration doesn't happen. We don't preemptively
  cancel a mid-stream generation.
- **First iteration always runs**, no matter the budget — the check can't
  fire until `TurnMetrics` lands.
- **User turns are subject to the same budget.** The old 16-iteration
  protection was a hard cap for user turns too; now they can spin up to
  the token budget's limit before erroring. Ordinarily fine (500k default
  is generous), worth noting because a broken user-triggered tool loop
  could now run much longer before the safety net catches it.
- **Compaction is excluded.** The summarizer call doesn't emit
  `TurnMetrics`, so it doesn't count against the turn's budget.

### Docs updates

- [docs/sessions.md](docs/sessions.md) — new "Turn token budget" section;
  error-code table adds `token_budget_exceeded`.
- [docs/projects.md](docs/projects.md) — cron section updated to cover
  `max_tokens`.
- [docs/config.md](docs/config.md) — agent-field table adds
  `max_turn_tokens`.

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
