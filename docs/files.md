# File Transfer

Two flows: clients can upload files into an agent's workspace, and agents can
hand files back to clients. Bytes move over REST; coordination rides the
existing WebSocket and message history.

For session creation, history, per-session context, and the WebSocket
protocol overview, see [sessions.md](sessions.md). For the higher-level
**project** abstraction (shared user-visible working directories), see
[projects.md](projects.md). The browsable / editable REST surface for
agent workspaces is documented below under
[Workspace filesystem API](#workspace-filesystem-api).

## Where files live

Every agent has a workspace directory ([docs/config.md](config.md)). Within
it, two conventions:

- `<workspace>/uploads/` — a **shared bucket** where every client-uploaded
  file lands. All of the agent's sessions see this directory. Filename
  collisions auto-suffix: `report.pdf` → `report-2.pdf` → `report-3.pdf`. The
  agent disambiguates by mtime (newest first via `list_uploads`).
- The rest of the workspace is the agent's own scratch space. Anything the
  agent writes there is available to share back to clients.

Uploads survive session deletion; they belong to the agent, not the session.
If you want them gone, delete the file from `uploads/` directly (e.g. via the
agent's `run_command` tool or out-of-band).

## REST endpoints

All endpoints require `Authorization: Bearer <auth_secret>`. Paths in the
table below are relative to the server's base URL (e.g.
`http://ark-ds:7777`).

### Upload

```
POST /agents/{name}/sessions/{sid}/uploads
Content-Type: multipart/form-data
Body: a single form field "file" containing the file bytes
```

Limits:
- 25 MB hard cap (server returns `413` if exceeded; partial file is removed)
- One file per request

Response:
```json
{
  "path": "uploads/report.pdf",       // workspace-relative
  "original_name": "report.pdf",      // the filename before any auto-suffix
  "size": 84211
}
```

Side effects:
- Bytes streamed to `<workspace>/uploads/<filename>` (auto-suffixed if needed)
- An `UploadMessage` is appended to the session's message history, so the
  conversation transcript records what was attached
- On the model's next turn, an `[user attached a file: …]` marker is folded
  into the prompt so the LLM knows it exists

Errors: `404` (unknown agent/session), `400` (missing filename, bad path),
`413` (over the size cap), `401` (bad bearer token).

### List uploads

```
GET /agents/{name}/sessions/{sid}/uploads
```

Returns the contents of the shared `uploads/` bucket, **newest first**:

```json
[
  {"path": "uploads/report-2.pdf", "size": 50121, "mtime": 1778702633},
  {"path": "uploads/report.pdf",   "size": 48902, "mtime": 1778702520}
]
```

Note: `sid` is required in the URL for routing, but the response is the same
across sessions of one agent — the bucket is shared, not session-scoped.

### Download

```
GET /agents/{name}/files/{workspace-relative-path}
```

Streams any file inside the agent's workspace. Common use:
- Fetch an upload: `/agents/scribe/files/uploads/report.pdf`
- Fetch something the agent wrote: `/agents/scribe/files/chart.png`

Path traversal is blocked at the resolver — `..` segments, encoded variants
(`..%2F`, `%2e%2e%2F`, …), absolute paths, and symlinks that escape the
workspace all return `400`. Files that don't exist return `404`.

Response: the file bytes, `Content-Type` guessed from the filename,
`Content-Disposition: attachment; filename="…"`.

## WebSocket event: `file_available`

When the agent calls `share_with_client(path, description?)`, the server
publishes a WebSocket event to every client connected to that session:

```json
{
  "type": "file_available",
  "path": "chart.png",
  "description": "Q4 summary",
  "size": 84211,
  "root": "workspace"
}
```

`path` is relative to `root`. For `root: "workspace"` (the default),
fetch via `GET /agents/{name}/files/<path>` as above. For
`root: "project"` the frame also carries `project_id`, and clients
fetch via `GET /projects/{project_id}/files/<path>`:

```json
{
  "type": "file_available",
  "path": "Q3-report.pdf",
  "description": "quarterly",
  "size": 262144,
  "root": "project",
  "project_id": "abc-123"
}
```

The event is emitted *after* the corresponding `SharedFile` message has been
persisted to history, so clients that connect later still see it via
`GET /history`.

## Agent-side tools

The agent has two built-in tools for file transfer:

| Tool | Purpose |
|---|---|
| `list_uploads()` | List files in the agent's `uploads/` directory, newest first. Convenience around `list_files("uploads")` that the agent gets up-front in the manifest. |
| `share_with_client(path, description="")` | Make a file downloadable by the client. Accepts (a) a workspace-relative path, or (b) an absolute path pointing into either the agent's workspace OR the current session's bound project root. The file is not copied — it stays where it is and the client fetches it directly. Persists a `SharedFile` message (with `root: "workspace"` or `root: "project"`) and publishes `file_available` with the matching `root`/`project_id`. Paths outside both locations are rejected. |

The system prompt automatically tells the agent about these tools and where
attachments live ([ark/runtime.py:system_prompt](../ark/runtime.py)). Agents
don't need to be reminded by the user.

## CLI usage

```
# Upload files before opening the chat (repeatable)
ark chat scribe --attach ~/Documents/report.pdf --attach ~/data/q4.csv

# Or attach mid-chat
you> /attach ~/Downloads/screenshot.png

# When the agent shares a file back, it's auto-downloaded:
[agent shared chart.png (84211 bytes) → ark-downloads/chart.png — Q4 summary]

# Customize the download location
ark chat scribe --download-dir ~/ark-out
```

If `chart.png` already exists in the download dir, the local copy is
auto-suffixed (`chart-2.png`) — same pattern as server-side upload
collisions, but applied locally.

## History representation

Two new message kinds appear in `GET /agents/{name}/sessions/{sid}/history`:

```json
{ "kind": "UploadMessage",
  "data": { "path": "uploads/report.pdf", "original_name": "report.pdf", "size": 48902 } }

{ "kind": "SharedFile",
  "data": { "path": "chart.png", "description": "Q4 summary", "size": 84211 } }

{ "kind": "SharedFile",
  "data": { "path": "Q3-report.pdf", "description": "quarterly",
            "size": 262144, "root": "project", "project_id": "abc-123" } }
```

`root` and `project_id` are omitted from the on-disk row for the
workspace-default case (keeping legacy rows tiny); loaders fill the
defaults (`root: "workspace"`, `project_id: null`) automatically.

When constructing the next LLM prompt, both providers fold these into text
markers in the natural place (user-side text for uploads, assistant-side
text for shared files), so the model sees a complete and coherent transcript.

## Things that aren't supported in v1

- **Chunked / resumable uploads.** One shot, max 25 MB.
- **Sending file *contents* to the model.** The model is told a file exists
  and where; it must call `read_file` to look at the bytes. Native
  multimodal upload (vision, PDF) is a separate, larger piece.
- **Per-file permissions.** Anything in the workspace is downloadable by any
  client with the bearer token. The token is the only credential.
- **Server-initiated push.** Files only become visible to clients when the
  agent explicitly calls `share_with_client`. There's no inbox-watching.

## Workspace filesystem API

In addition to the session-scoped upload flow above, every agent's workspace
is browsable / editable directly over REST. Same shape as the per-project
filesystem API ([projects.md](projects.md)); same path-traversal
protections; same live `file_changed` events.

```
GET    /agents/{name}/files                 # list workspace root
GET    /agents/{name}/files/{path}          # file → bytes; dir → JSON listing
PUT    /agents/{name}/files/{path}          # write file (raw body)
DELETE /agents/{name}/files/{path}          # delete file or directory (recursive)
POST   /agents/{name}/files/{path}?op=mkdir
POST   /agents/{name}/files/{path}?op=rename&dest=<dest>   # move/rename
```

`op=rename` behavior (mirrors the project endpoint):

- `path` is the source, `dest` is the new location — both workspace-relative.
- Works on files and directories (moving a directory carries its subtree).
- Parent directories of `dest` are created if missing.
- `404` if the source doesn't exist; `409` if `dest` already exists.
- Path-traversal checks apply to both source and destination.

Listings return:

```json
{
  "path": "scratch",
  "entries": [
    {"name": "draft.md", "is_dir": false, "size": 1234, "mtime": 1779000000000},
    {"name": "subdir",   "is_dir": true,  "size": 0,    "mtime": 1779000000000}
  ]
}
```

Path traversal is enforced — `..`, absolute paths, encoded variants, and
symlinks that escape the workspace root all return `400`.

### Live workspace change events

Each agent's workspace is also watched for filesystem changes. Events fan
out over the unified `/events` WS as:

```json
{
  "type": "workspace_file_changed",
  "agent_name": "scribe",
  "path": "scratch/draft.md",
  "change": "created" | "modified" | "deleted"
}
```

Same coalescing window (~200ms) and ignore-list (`.git`, `node_modules`,
`__pycache__`, `.pytest_cache`, `.venv`, `.DS_Store`, `.idea`, `.vscode`)
as `project_file_changed`. Move events surface as a delete on the source
path followed by a create on the destination.

### When to use workspace vs. project endpoints

| | Workspace | Project |
|---|---|---|
| Scope | One agent's private scratch space | Shared across one or more sessions / agents |
| User-visible "thing" | Usually no — debugging or inspection | Yes — the user's working artifact |
| Default upload destination | When session is **not** in a project | When session **is** in a project |
| Live change events | `workspace_file_changed`, keyed by `agent_name` | `project_file_changed`, keyed by `project_id` |

In short: projects are first-class collaboration surfaces with their own
lifecycle (`POST /projects`, soft-delete, etc.); the workspace endpoints
exist so you can also poke around an agent's private space when you need
to.

## Security notes

- Path traversal is enforced in [ark/workspace.py:resolve](../ark/workspace.py).
  See `tests/test_file_transfer.py::test_rest_download_never_leaks_external_file`
  for the invariant — an arbitrary URL must never return bytes from outside
  the workspace.
- The 25 MB cap is enforced during stream-to-disk; partial bytes from an
  oversized upload are removed before the 413 is returned.
- The bearer token is the only access control. Anyone with the token can
  download anything in any agent's workspace. Operate accordingly — see
  [deploy.md](deploy.md) for hardening notes (TLS, firewall, rotation).
