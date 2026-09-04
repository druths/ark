# Projects

A **project** is a shared, user-visible working directory that one or more
sessions can be bound to. Unlike an agent's workspace (per-agent, private),
a project is intended for files the user inspects, edits, uploads to, and
watches changing in real time.

Typical uses:

- A document the agent is drafting (the agent writes, the user edits, both
  iterate).
- A code repository the agent is reasoning over.
- A shared scratch space for a long-running task where the final artifacts
  matter more than the chat transcript.

Projects are persistent and survive across sessions, restarts, and (softly)
deletion. The agent's workspace is unchanged — when a session is in a
project, the workspace still exists as private scratch space; the project
is just the default working surface.

## Lifecycle

```
POST   /projects                  # create
GET    /projects                  # list active; ?include_deleted=true for all
GET    /projects/{id}             # fetch one
PUT    /projects/{id}             # update name/description/project_context
DELETE /projects/{id}             # soft-delete (files survive)
```

### Create

```json
POST /projects
{
  "name": "marketing-brochure",
  "root": "/var/lib/ark/projects/abc-123",    // optional; default below
  "description": "Q4 product brochure draft",  // optional
  "project_context": "Tone: warm, professional.\nAudience: enterprise."   // optional
}
```

Defaults:

- If `root` is omitted, the directory is created at
  `<ARK_HOME>/projects/<id>/`.
- If `root` points at an existing directory, the project adopts it as-is.
- If `root` points at a path that doesn't exist, the directory is created.

Names must be unique among active projects. A soft-deleted project's name
becomes available again.

### Soft-delete

`DELETE /projects/{id}` flips `deleted_at` on the row. **Nothing on disk is
touched.** Sessions that were bound to the project retain their
`project_id` column for history continuity, but `runtime.session_project()`
returns `None` for them so the runtime stops layering project context.

To "really" delete a project, do the row deletion via SQL plus `rm -rf` on
the directory. There is intentionally no API for that.

## Per-project filesystem API

Each project has a filesystem rooted at its `root`. Clients can browse,
read, write, and delete files through these endpoints:

```
GET    /projects/{id}/files                  # list root contents
GET    /projects/{id}/files/{path}           # file: stream bytes; dir: list JSON
PUT    /projects/{id}/files/{path}           # write file (raw body)
DELETE /projects/{id}/files/{path}           # delete file or directory (recursive)
POST   /projects/{id}/files/{path}?op=mkdir
POST   /projects/{id}/files/{path}?op=rename&dest=<dest>   # move/rename
```

`op=rename` behavior:

- `path` is the source, `dest` is the new location — both project-relative.
- Works on files and directories (moving a directory carries its subtree).
- Parent directories of `dest` are created if missing.
- `404` if the source doesn't exist; `409` if `dest` already exists (no
  silent overwrite).
- `dest` is path-traversal checked the same way `path` is — any attempt to
  rename out of the project root returns `400`.

Listings return:

```json
{
  "path": "subdir",
  "entries": [
    {"name": "draft.md", "is_dir": false, "size": 1234, "mtime": 1779000000000},
    {"name": "images",   "is_dir": true,  "size": 0,    "mtime": 1779000000000}
  ]
}
```

Path traversal is enforced — `..`, absolute paths, encoded variants, and
symlinks that escape the root all return `400`.

## Binding a session to a project

Pass `project_id` when creating the session:

```json
POST /agents/{name}/sessions
{
  "project_id": "abc-123",         // optional; null = workspace-only session
  "context": "..."                  // existing per-session context option
}
```

The binding can be changed at any time via `PATCH .../project` (see
[Reassigning a session's project](#reassigning-a-sessions-project) below).

When a session is in a project, the runtime adjusts three things:

1. **System prompt** gains a Project section between the Environment stanza
   and any per-session context:

   ```
   Project (this session):
   - Name: marketing-brochure
   - Root: /var/lib/ark/projects/abc-123
   - Description: Q4 product brochure draft

   Tone: warm, professional.
   Audience: enterprise.

   All file operations should target paths under the project root above
   unless explicitly asked to modify your workspace. The project is where
   the user can see and edit your work; your workspace is private scratch
   space. Uploads in this session land in `<project_root>/uploads/`.
   ```

2. **Uploads** (`POST /agents/{name}/sessions/{sid}/uploads`) land in
   `<project_root>/uploads/` instead of the workspace. `list_uploads` (REST
   and agent tool) lists the same directory.

3. **`cwd` is unchanged**. The agent's working directory is still its
   workspace — by design, for predictability. The agent uses absolute paths
   for project file ops, and that's what the system prompt directs it to
   do.

## Reassigning a session's project

The project binding is mutable. A session can be attached to a project,
moved to a different one, or detached entirely.

```
PATCH /agents/{name}/sessions/{sid}/project
Body: { "project_id": "<uuid>" }     # reassign (or first-time assign)
      { "project_id": null }          # detach

200: { "ok": true, "changed": true, "from": {id,name,root}|null, "to": {id,name,root}|null }
200: { "ok": true, "changed": false }         # no-op (already assigned as requested)
404: unknown agent, session, or project (soft-deleted target counts as unknown)
409: session has unmatched tool calls — wait for the turn to complete
400: body missing 'project_id', or project_id is not a string or null
```

On a real change (`changed: true`), two things happen:

1. **A `ProjectAssignmentChanged` marker is persisted** in the session's
   history. `GET /history` returns it so clients can render a "── Project
   changed to Y ──" divider in the transcript. Its payload records
   `from_project_*` and `to_project_*` (id, name, root) plus `changed_at`.
   Successive reassignments produce successive markers — an ordered
   assignment history.
2. **A `session_project_changed` event is published** on `/events` so file
   browsers and other live UIs can refresh. Payload mirrors the marker.

On the **next turn**, the runtime substitutes the marker with a synthetic
`UserText` notification in the message list — the LLM sees an explicit
event describing the transition (previous project, new project, and a
reminder that prior file references are historical context). The system
prompt's "Project" stanza automatically reflects the new binding. Nothing
about the old project's context or `project_context` carries forward.

**No-op semantics.** If the requested `project_id` equals the current one,
the endpoint returns `{"changed": false}` and does not write a marker or
publish an event — clients that don't want to check-before-set can PATCH
idempotently.

**Uploads before/after the change.** Files uploaded while the session was
in project A live under `/projects/A/uploads/`. After reassignment to
project B, `list_uploads` (agent tool + REST) reflects project B's dir —
the old uploads still exist on disk but are no longer surfaced to the
agent through `list_uploads`. The transition notification in the LLM
message list explicitly warns that references to prior project files are
historical.

**Compaction interaction.** The default summarizer prompt already
preserves significant events, so a compaction that spans a reassignment
should carry the transition forward in its summary. If it doesn't in
practice, the client can pre-supply a summary via `POST .../compact` that
includes the reassignment context.

### CLI

```
ark session set-project <sid> <project-name>     # reassign to that project by name
ark session set-project <sid> --none              # detach
```

The command resolves the session's owning agent from its metadata and
looks up the project id by name via `GET /projects`.

## Agent tools

| Tool | Behavior |
|---|---|
| `get_current_session_info()` | Now includes `project_id`, `project_name`, `project_root` (null when the session isn't in a project). |
| `get_project_info()` | Returns the project record (id, name, root, description, project_context) or null. Use when you want richer project metadata. |
| `list_uploads()` | Lists `<project_root>/uploads/` when in a project, `<workspace>/uploads/` otherwise. |
| `read_file` / `write_file` / `list_files` / `run_command` | Unchanged. The agent uses absolute paths to operate on project files. |

## Live file-change events

When a client is connected to the unified `/events` WebSocket, it also
receives:

```json
{
  "type": "project_file_changed",
  "project_id": "abc-123",
  "path": "subdir/draft.md",
  "change": "created" | "modified" | "deleted"
}
```

These come from a `watchdog`-based per-project filesystem watcher. Events
are coalesced within a ~200ms window to collapse editor-save bursts. The
following paths are ignored by default to keep noise down: `.git`,
`node_modules`, `__pycache__`, `.pytest_cache`, `.venv`, `.DS_Store`,
`.idea`, `.vscode`.

Move events surface as a delete on the source path followed by a create on
the destination.

## Multiple agents in one project

Projects are not coupled to agents. You can bind sessions from `scribe`,
`editor`, and `reviewer` all to the same project, and they'll all see the
same root, system-prompt section, and upload location. This is intended —
projects represent shared artifacts, not agent-private territory.

## Cron entries can be bound to a project

Each cron entry has an optional `project_id`. When set, every fire of that
cron creates a session already bound to the project — the system prompt
includes the "Project" stanza from turn 1, and uploads land in the
project's `uploads/`.

**REST:**

```
PUT /agents/{name}/crons/{cron_id}
Body: {
  "expr": "...",
  "prompt": "...",
  "project_id"?: "<uuid>" | null,
  "max_tokens"?: <positive int> | null
}
```

Both `project_id` and `max_tokens` are optional and use the same
"omit = preserve, null = clear" semantics: omitting on update keeps the
existing value; explicit `null` explicitly reverts (detaches the project /
drops the per-cron budget override).

- Unknown or soft-deleted project → `404`.
- Non-positive-integer `max_tokens` → `400`.

`max_tokens` sets a per-cron override for the per-turn token budget (see
[sessions.md § Turn token budget](sessions.md#turn-token-budget)). Useful
when a specific cron does heavier work than the agent's default budget
allows, or when you want to cap a specific cron more tightly than the
agent's default.

`GET /agents/{name}/crons` returns `project_id` and `project_name` on each
row. A cron whose bound project was soft-deleted returns `project_name:
null` (the id survives for audit).

**Agent tool:**

```python
add_cron(id, expr, prompt, project_id?, max_tokens?)
```

To discover a project's id, agents can call the new `list_projects` tool
(active projects only), or `get_current_session_info` if their current
session is already in the project they want. `max_tokens` is the same
per-cron budget override as the REST endpoint.

**Scheduler behavior when the bound project is soft-deleted between
definition and fire time:** the scheduler logs a warning to stderr
(`[scheduler] cron X for agent Y: bound project Z is deleted, firing in
workspace mode`), still fires the cron, and records the dangling
`project_id` on the session row for audit. `runtime.session_project()`
returns `None` for the deleted project, so the session runs project-less
(no Project stanza in the system prompt, uploads back to the workspace).

**CLI:**

```
ark cron set <agent> <id> "<expr>" --prompt "..." --project <name>     # bind project
ark cron set <agent> <id> "<expr>" --prompt "..." --no-project          # detach project
ark cron set <agent> <id> "<expr>" --prompt "..." --max-tokens 2000000  # per-cron budget
ark cron set <agent> <id> "<expr>" --prompt "..." --no-max-tokens       # revert to agent/global default
ark cron set <agent> <id> "<expr>" --prompt "..."                       # keep whatever's already set
ark cron list <agent>                                                    # shows [project=<name>] and [max_tokens=N]
```

Sharp edges:

- **No enforcement of project consistency across an agent's crons.** One
  agent can have crons bound to different projects (or none at all)
  simultaneously. This is a feature — a single "operations" agent can run
  scheduled work across multiple projects.
- **No connection to a user's conversational session** in the same
  project. Cron-fired sessions and user sessions are independent even when
  both are in the same project. Use `post_to_session` if a cron needs to
  hand its output to a conversational session.

## What v1 doesn't cover

- **Heartbeats aren't project-bound.** They're agent-level scheduling —
  the "check on things you own" tick. If a heartbeat needs to touch a
  project, its `heartbeat_prompt.md` can mention the project root
  explicitly.
- **Concurrent-edit locking.** If the agent is writing a file via
  `write_file` while the client is editing it via `PUT /files/...`, last
  write wins. No conflict detection in v1.
- **Per-project event filtering.** All connected `/events` clients hear
  about all project changes. Add client-side filtering on `project_id` if
  you only want a subset. Server-side per-project subscription is a
  possible future addition.
- **Restricted project roots.** A project root can point anywhere on the
  server filesystem. The Ark server runs as root in the intended
  deployment, so this is expected. Be deliberate.
