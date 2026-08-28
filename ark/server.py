"""FastAPI server: REST + WebSocket session interaction + scheduler lifecycle."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import broker, db, file_watcher, mcp, projects, runtime, skills, workspace as ws
from .config import Config
from .scheduler import Scheduler
from .types import AssistantText, UploadMessage, message_from_row

UPLOAD_MAX_BYTES = 25 * 1024 * 1024  # 25 MB v1 cap


def create_app(config: Config) -> FastAPI:
    conn = db.init_db()
    skills.discover(list(config.agents))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        sched = Scheduler(conn, config)
        sched.start()
        watcher = file_watcher.get_watcher()
        await watcher.start()
        # Watch every active project that already exists.
        for p in projects.list_projects(conn):
            watcher.watch("project", p.id, p.root)
        # Watch every agent's workspace so REST file edits surface as live events.
        for agent in config.agents.values():
            agent.workspace.mkdir(parents=True, exist_ok=True)
            watcher.watch("workspace", agent.name, str(agent.workspace))
        # Connect to configured MCP servers. Per-server failures don't abort
        # startup — the manager records them and affected tools error at call.
        mcp_manager = mcp.get_manager()
        await mcp_manager.start(config.mcp_servers)
        try:
            yield
        finally:
            await mcp_manager.stop()
            await watcher.stop()
            await sched.stop()

    app = FastAPI(title="Ark", lifespan=lifespan)
    app.state.config = config
    app.state.conn = conn

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if not _check_bearer(request.headers.get("authorization", ""), config.server.auth_secret):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/agents")
    def list_agents():
        return [
            {
                "name": a.name,
                "provider": a.provider,
                "provider_type": config.providers[a.provider].provider_type,
                "model": a.model,
                "workspace": str(a.workspace),
            }
            for a in config.agents.values()
        ]

    @app.get("/agents/{name}")
    def get_agent(name: str):
        agent = config.agents.get(name)
        if agent is None:
            raise HTTPException(404, "unknown agent")
        provider_cfg = config.providers[agent.provider]
        heartbeat = conn.execute(
            "SELECT heartbeat_seconds FROM agent_state WHERE agent_name = ?", (name,)
        ).fetchone()
        crons = conn.execute(
            "SELECT id, expr, prompt, project_id FROM crons WHERE agent_name = ? AND enabled = 1",
            (name,),
        ).fetchall()
        mcp_manager = mcp.get_manager()
        mcp_status = []
        for server_name in agent.mcp_servers:
            srv = mcp_manager.get(server_name)
            mcp_status.append(
                {
                    "name": server_name,
                    "ready": srv is not None and srv.error is None,
                    "error": srv.error if srv is not None else "not started",
                    "tool_count": len(srv.tools) if srv is not None else 0,
                    "always_loaded": server_name in agent.always_loaded_mcp_servers,
                }
            )
        return {
            "name": agent.name,
            "provider": agent.provider,
            "provider_type": provider_cfg.provider_type,
            "model": agent.model,
            "workspace": str(agent.workspace),
            "always_loaded_skills": agent.always_loaded_skills,
            "mcp_servers": mcp_status,
            "heartbeat_seconds": heartbeat["heartbeat_seconds"] if heartbeat else None,
            "crons": [dict(r) for r in crons],
        }

    @app.put("/agents/{name}/heartbeat")
    async def set_heartbeat(name: str, body: dict):
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        seconds = body.get("interval_seconds")
        if seconds is None:
            conn.execute(
                "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, NULL) "
                "ON CONFLICT(agent_name) DO UPDATE SET heartbeat_seconds = NULL",
                (name,),
            )
        else:
            conn.execute(
                "INSERT INTO agent_state(agent_name, heartbeat_seconds) VALUES (?, ?) "
                "ON CONFLICT(agent_name) DO UPDATE SET heartbeat_seconds = excluded.heartbeat_seconds",
                (name, int(seconds)),
            )
        return {"ok": True}

    @app.get("/agents/{name}/crons/{cron_id}/sessions")
    def list_cron_fires(name: str, cron_id: str, limit: int = 20):
        """List the past fires (sessions) of a specific cron entry, newest
        first. Enriched with a one-line summary + error metadata so clients
        can render a tabular log without a round-trip per row."""

        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        limit = max(1, min(int(limit), 200))
        rows = conn.execute(
            "SELECT id, created_at, ended_at FROM sessions "
            "WHERE agent_name = ? AND cron_id = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (name, cron_id, limit),
        ).fetchall()
        out = []
        for r in rows:
            sid = r["id"]
            # First post_to_session call's body is the most useful summary
            # for the common "cron messages the user" pattern. Otherwise the
            # last assistant text. Otherwise "(no output)".
            summary = "(no output)"
            for tc in conn.execute(
                "SELECT content_json FROM messages WHERE session_id = ? "
                "AND role = 'tool_call' ORDER BY id",
                (sid,),
            ):
                c = json.loads(tc["content_json"])
                if c.get("name") == "post_to_session":
                    body = (c.get("input") or {}).get("body") or ""
                    if body:
                        summary = body
                        break
            if summary == "(no output)":
                last_text = conn.execute(
                    "SELECT content_json FROM messages WHERE session_id = ? "
                    "AND role = 'assistant' ORDER BY id DESC LIMIT 1",
                    (sid,),
                ).fetchone()
                if last_text is not None:
                    txt = json.loads(last_text["content_json"]).get("text") or ""
                    if txt:
                        summary = txt
            # Error info — pull the most recent run_error row if any.
            err_row = conn.execute(
                "SELECT content_json FROM messages WHERE session_id = ? "
                "AND role = 'run_error' ORDER BY id DESC LIMIT 1",
                (sid,),
            ).fetchone()
            had_error = err_row is not None
            error_code = None
            if had_error:
                error_code = json.loads(err_row["content_json"]).get("code")
            out.append(
                {
                    "session_id": sid,
                    "created_at": r["created_at"],
                    "ended_at": r["ended_at"],
                    "had_error": had_error,
                    "error_code": error_code,
                    "summary": summary[:300],
                }
            )
        return out

    @app.get("/agents/{name}/crons")
    def list_crons(name: str):
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        rows = conn.execute(
            "SELECT id, expr, prompt, enabled, project_id "
            "FROM crons WHERE agent_name = ? ORDER BY id",
            (name,),
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            d["project_name"] = None
            if d["project_id"]:
                p = projects.get(conn, d["project_id"])
                # Show name even for soft-deleted so the operator can see WHAT
                # was bound and understand why fires are running project-less.
                d["project_name"] = p.name if p is not None else None
            out.append(d)
        return out

    @app.put("/agents/{name}/crons/{cron_id}")
    def upsert_cron(name: str, cron_id: str, body: dict):
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        expr = body.get("expr")
        prompt = body.get("prompt")
        if not expr or not prompt:
            raise HTTPException(400, "expr and prompt are required")
        try:
            from croniter import croniter

            croniter(expr)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"invalid cron expression: {e}")
        # `project_id` is optional. When present it must be a string uuid of
        # an existing, non-soft-deleted project; explicit null detaches.
        # Omitted → keep whatever was there on update, or null on insert.
        project_id_present = "project_id" in body
        project_id = body.get("project_id")
        if project_id_present and project_id is not None:
            if not isinstance(project_id, str):
                raise HTTPException(400, "'project_id' must be a string or null")
            p = projects.get(conn, project_id)
            if p is None or p.deleted_at is not None:
                raise HTTPException(404, "unknown project")
        if project_id_present:
            conn.execute(
                "INSERT INTO crons(agent_name, id, expr, prompt, enabled, project_id) "
                "VALUES (?,?,?,?,1,?) "
                "ON CONFLICT(agent_name, id) DO UPDATE SET "
                "expr=excluded.expr, prompt=excluded.prompt, enabled=1, "
                "project_id=excluded.project_id",
                (name, cron_id, expr, prompt, project_id),
            )
        else:
            # Omitted → don't touch project_id on existing rows; keep null on new.
            conn.execute(
                "INSERT INTO crons(agent_name, id, expr, prompt, enabled) VALUES (?,?,?,?,1) "
                "ON CONFLICT(agent_name, id) DO UPDATE SET "
                "expr=excluded.expr, prompt=excluded.prompt, enabled=1",
                (name, cron_id, expr, prompt),
            )
        return {"ok": True}

    @app.delete("/agents/{name}/crons/{cron_id}")
    def delete_cron(name: str, cron_id: str):
        cur = conn.execute(
            "DELETE FROM crons WHERE agent_name = ? AND id = ?", (name, cron_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "unknown cron")
        return {"ok": True}

    @app.get("/agents/{name}/sessions")
    def list_sessions(name: str, kind: str | None = None, limit: int = 50):
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        return runtime.list_sessions(conn, name, kind=kind, limit=limit)

    @app.post("/agents/{name}/sessions")
    async def create_session(name: str, request: Request):
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        # Body is optional. Accept an empty request, or `{context?, project_id?}`.
        ctx_text = ""
        project_id: str | None = None
        try:
            body = await request.json()
            if isinstance(body, dict):
                ctx_text = (body.get("context") or "").strip()
                project_id = body.get("project_id") or None
        except Exception:
            pass  # no/invalid body → just create the session

        if project_id is not None:
            p = projects.get(conn, project_id)
            if p is None or p.deleted_at is not None:
                raise HTTPException(400, f"unknown or deleted project: {project_id!r}")

        sid = runtime.create_session(
            conn, name, "conversational", project_id=project_id
        )
        if ctx_text:
            runtime.append_context(conn, sid, ctx_text)
        return {"id": sid}

    @app.post("/agents/{name}/sessions/{sid}/context")
    async def append_session_context(name: str, sid: str, request: Request):
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "expected JSON body")
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        text = (body.get("context") or "").strip()
        if not text:
            raise HTTPException(400, "missing or empty 'context'")
        raw_name = body.get("name")
        block_name = str(raw_name).strip() or None if raw_name is not None else None
        count, replaced = runtime.append_context(conn, sid, text, name=block_name)
        return {"ok": True, "count": count, "replaced": replaced}

    @app.delete("/agents/{name}/sessions/{sid}")
    def delete_session(name: str, sid: str):
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        runtime.delete_session(conn, sid)
        return {"ok": True}

    @app.patch("/agents/{name}/sessions/{sid}/project")
    async def set_session_project_endpoint(name: str, sid: str, request: Request):
        """Reassign or detach a session's project binding.

        Body: `{"project_id": "<uuid>" | null}`
        - non-null → reassign to that project (or no-op if already there)
        - null → detach

        Persists a ProjectAssignmentChanged marker in history (the LLM sees a
        synthetic notification of the transition on the next turn); publishes
        a `session_project_changed` event so file-browser UIs can refresh.

        Guards: 404 unknown agent/session/project (or soft-deleted target);
        409 if the session has pending tool calls. Idempotent no-op returns
        200 with `{"ok": true, "changed": false}`.
        """
        if name not in config.agents:
            raise HTTPException(404, "unknown agent")
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "expected JSON body")
        if not isinstance(body, dict) or "project_id" not in body:
            raise HTTPException(400, "body must be an object with a 'project_id' field")
        new_project_id = body["project_id"]
        if new_project_id is not None and not isinstance(new_project_id, str):
            raise HTTPException(400, "'project_id' must be a string or null")
        if new_project_id is not None:
            target = projects.get(conn, new_project_id)
            if target is None or target.deleted_at is not None:
                raise HTTPException(404, "unknown project")
        if runtime.has_pending_tool_calls(runtime.load_history(conn, sid)):
            raise HTTPException(
                409, "session has unmatched tool calls — wait for the turn to complete"
            )

        result = runtime.set_session_project(conn, sid, new_project_id)
        if result is None:
            return {"ok": True, "changed": False}
        from_p, to_p = result

        def _proj_dict(p):
            if p is None:
                return None
            return {"id": p.id, "name": p.name, "root": p.root}

        broker.publish(
            sid,
            {
                "type": "session_project_changed",
                "session_id": sid,
                "agent_name": name,
                "from_project_id": from_p.id if from_p else None,
                "from_project_name": from_p.name if from_p else None,
                "to_project_id": to_p.id if to_p else None,
                "to_project_name": to_p.name if to_p else None,
                "changed_at": runtime.now_ms(),
            },
        )
        return {
            "ok": True,
            "changed": True,
            "from": _proj_dict(from_p),
            "to": _proj_dict(to_p),
        }

    @app.post("/agents/{name}/sessions/{sid}/compact")
    async def compact_session_endpoint(name: str, sid: str, request: Request):
        """Manually trigger compaction for a session.

        Body is optional:
        - `{}` or empty: server generates the summary via the session's own
          provider (same summarizer as auto-triggered compactions).
        - `{"summary": "..."}`: use the supplied text verbatim, no LLM call.

        Same lifecycle events fire on `/events` as automatic compactions, so
        any connected WS client sees the work happening. Respects the same
        pending-tool-call guard as reactive compaction: if the session is
        mid-tool-loop, returns 409. `compaction_enabled=false` on the agent
        does NOT gate this endpoint — the client is explicitly asking.
        """
        agent = config.agents.get(name)
        if agent is None:
            raise HTTPException(404, "unknown agent")
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")

        history = runtime.load_history(conn, sid)
        if runtime.has_pending_tool_calls(history):
            raise HTTPException(
                409, "session has unmatched tool calls — wait for the turn to complete"
            )

        # Empty body is acceptable → server-generated.
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")

        supplied = body.get("summary")
        if supplied is not None:
            if not isinstance(supplied, str) or not supplied.strip():
                raise HTTPException(400, "'summary' must be a non-empty string when provided")
            # Client-supplied path: skip the LLM call entirely. Still publish
            # the lifecycle events so WS clients see the same shape.
            reason = "client-supplied"
            started = {
                "type": "compaction_started",
                "session_id": sid, "agent_name": name,
                "reason": reason, "input_tokens": None,
                "context_window": None, "model": agent.model,
            }
            broker.publish(sid, started)
            from .types import CompactionSummary
            runtime.append_message(
                conn, sid, CompactionSummary(text=supplied.strip(), reason=reason)
            )
            completed = {
                "type": "compaction_completed",
                "session_id": sid, "agent_name": name,
                "summary": supplied.strip(), "reason": reason,
            }
            broker.publish(sid, completed)
            return {"ok": True, "summary": supplied.strip(), "reason": reason}

        # Server-generated: drive compact_session and publish each event.
        summary_text = ""
        fail_info: dict | None = None
        async for evt in runtime.compact_session(
            conn=conn, config=config, agent=agent, session_id=sid,
            reason="client-invoked",
        ):
            wire = runtime.event_to_wire(evt)
            wire["session_id"] = sid
            wire["agent_name"] = name
            broker.publish(sid, wire)
            # Capture terminal state for the HTTP response.
            from .types import (
                CompactionCompletedEvent as _CE,
                CompactionFailedEvent as _CF,
            )
            if isinstance(evt, _CE):
                summary_text = evt.summary
            elif isinstance(evt, _CF):
                fail_info = {"code": evt.code, "message": evt.message}
        if fail_info is not None:
            return JSONResponse(
                {"ok": False, "code": fail_info["code"], "message": fail_info["message"]},
                status_code=502,
            )
        return {"ok": True, "summary": summary_text, "reason": "client-invoked"}

    # ------------------------------------------------------------------
    # Projects: shared user-visible working directories
    # ------------------------------------------------------------------

    @app.post("/projects")
    async def create_project(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        try:
            p = projects.create(
                conn,
                name=body.get("name") or "",
                root=body.get("root"),
                description=body.get("description") or "",
                project_context=body.get("project_context") or "",
            )
        except projects.ProjectError as e:
            raise HTTPException(400, str(e))
        file_watcher.get_watcher().watch("project", p.id, p.root)
        return _project_to_json(p)

    @app.get("/projects")
    def list_projects_endpoint(include_deleted: bool = False):
        return [
            _project_to_json(p)
            for p in projects.list_projects(conn, include_deleted=include_deleted)
        ]

    @app.get("/projects/{pid}")
    def get_project(pid: str):
        p = projects.get(conn, pid)
        if p is None:
            raise HTTPException(404, "unknown project")
        return _project_to_json(p)

    @app.put("/projects/{pid}")
    async def update_project(pid: str, request: Request):
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, "expected JSON body")
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be an object")
        try:
            p = projects.update(
                conn,
                pid,
                name=body.get("name"),
                description=body.get("description"),
                project_context=body.get("project_context"),
            )
        except projects.ProjectError as e:
            raise HTTPException(400, str(e))
        return _project_to_json(p)

    @app.delete("/projects/{pid}")
    def delete_project(pid: str):
        # Soft-delete only — files on disk are untouched.
        if not projects.soft_delete(conn, pid):
            raise HTTPException(404, "unknown or already-deleted project")
        file_watcher.get_watcher().unwatch("project", pid)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Project filesystem endpoints
    # ------------------------------------------------------------------

    def _project_or_404(pid: str):
        p = projects.get(conn, pid)
        if p is None or p.deleted_at is not None:
            raise HTTPException(404, "unknown or deleted project")
        return p

    def _resolve_or_400(p, relative: str) -> Path:
        try:
            return projects.resolve_path(p, relative)
        except projects.ProjectPathError as e:
            raise HTTPException(400, str(e))

    def _dir_listing(target: Path, project_relative: str) -> dict:
        entries = []
        for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            st = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size": st.st_size if child.is_file() else 0,
                    "mtime": int(st.st_mtime * 1000),
                }
            )
        return {"path": project_relative, "entries": entries}

    @app.get("/projects/{pid}/files")
    def list_project_root(pid: str):
        p = _project_or_404(pid)
        root = Path(p.root)
        root.mkdir(parents=True, exist_ok=True)
        return _dir_listing(root, "")

    @app.get("/projects/{pid}/files/{path:path}")
    def get_project_path(pid: str, path: str):
        p = _project_or_404(pid)
        full = _resolve_or_400(p, path)
        if not full.exists():
            raise HTTPException(404, "not found")
        if full.is_dir():
            return _dir_listing(full, projects.relative_to_root(p, full))
        return FileResponse(full, filename=full.name)

    @app.put("/projects/{pid}/files/{path:path}")
    async def put_project_file(pid: str, path: str, request: Request):
        p = _project_or_404(pid)
        full = _resolve_or_400(p, path)
        full.parent.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        full.write_bytes(body)
        return {
            "ok": True,
            "path": projects.relative_to_root(p, full),
            "size": len(body),
        }

    @app.delete("/projects/{pid}/files/{path:path}")
    def delete_project_path(pid: str, path: str):
        p = _project_or_404(pid)
        full = _resolve_or_400(p, path)
        # Defense in depth: never let a malformed path resolve to the root
        # itself and wipe the project.
        if full == Path(p.root).resolve():
            raise HTTPException(400, "cannot delete the project root")
        if not full.exists():
            raise HTTPException(404, "not found")
        if full.is_dir():
            # Recursive removal — non-empty directories delete cleanly.
            import shutil

            shutil.rmtree(full)
        else:
            full.unlink()
        return {"ok": True}

    @app.post("/projects/{pid}/files/{path:path}")
    def post_project_path(pid: str, path: str, op: str = "", dest: str = ""):
        p = _project_or_404(pid)
        if op == "mkdir":
            full = _resolve_or_400(p, path)
            full.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "path": projects.relative_to_root(p, full)}
        if op == "rename":
            if not dest:
                raise HTTPException(400, "?dest=<path> is required for op=rename")
            src = _resolve_or_400(p, path)
            dst = _resolve_or_400(p, dest)
            if not src.exists():
                raise HTTPException(404, "source not found")
            if dst.exists():
                raise HTTPException(409, "destination already exists")
            dst.parent.mkdir(parents=True, exist_ok=True)
            # shutil.move handles cross-filesystem moves transparently;
            # Path.rename would raise EXDEV in that case.
            import shutil

            shutil.move(str(src), str(dst))
            return {
                "ok": True,
                "from": projects.relative_to_root(p, src),
                "to": projects.relative_to_root(p, dst),
            }
        raise HTTPException(400, f"unsupported op: {op!r} (try ?op=mkdir or ?op=rename)")

    @app.get("/sessions/{sid}")
    def get_session_metadata(sid: str):
        """Session metadata: id, agent_name, kind, timestamps, project + cron
        bindings. When this is a cron-kind session, also include the cron's
        prompt for transcript readability."""
        row = conn.execute(
            "SELECT id, agent_name, kind, created_at, ended_at, project_id, cron_id "
            "FROM sessions WHERE id = ?",
            (sid,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "unknown session")
        body = dict(row)
        if body["cron_id"]:
            cron_row = conn.execute(
                "SELECT prompt FROM crons WHERE agent_name = ? AND id = ?",
                (body["agent_name"], body["cron_id"]),
            ).fetchone()
            body["cron_prompt"] = cron_row["prompt"] if cron_row else None
        return body

    @app.get("/agents/{name}/sessions/{sid}/history")
    def get_history(name: str, sid: str):
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        return [_message_to_json(m) for m in runtime.load_history(conn, sid)]

    @app.get("/agents/{name}/sessions/{sid}/context")
    def get_session_context(name: str, sid: str):
        """Context introspection (mine-capstone#697): what the system prompt
        holds right now — its size, every context block (named blocks
        deduped exactly as the prompt builder renders them), the latest
        usage numbers and the compaction count."""
        agent = config.agents.get(name)
        if agent is None or not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        from .models import context_window_for
        from .types import CompactionSummary, SessionContext, TurnMetrics

        history = runtime.load_history(conn, sid)
        contexts = [m for m in history if isinstance(m, SessionContext)]
        latest = None
        for m in reversed(history):
            if isinstance(m, TurnMetrics):
                latest = m
                break
        compaction = None
        for m in reversed(history):
            if isinstance(m, CompactionSummary):
                compaction = m
                break
        project = runtime.session_project(conn, sid)
        system = runtime.system_prompt(
            agent, contexts, project=project,
            compaction_summary=compaction.text if compaction else None,
        )
        # Blocks exactly as the prompt renders them: named = last text at
        # first position; unnamed additive.
        latest_by_name = {c.name: c.text for c in contexts if c.name}
        seen: set[str] = set()
        blocks = []
        for seq, c in enumerate(contexts):
            if c.name:
                if c.name in seen:
                    continue
                seen.add(c.name)
                text = latest_by_name[c.name]
            else:
                text = c.text
            blocks.append({
                "name": c.name,
                "bytes": len(text.encode("utf-8")),
                "seq": seq,
            })
        return {
            "system_prompt_bytes": len(system.encode("utf-8")),
            "blocks": blocks,
            "last_input_tokens": latest.input_tokens if latest else None,
            "last_cached_input_tokens": latest.cached_input_tokens if latest else None,
            "context_window": context_window_for(agent.model, agent.max_context_tokens),
            "compactions": sum(
                1 for m in history if isinstance(m, CompactionSummary)
            ),
        }

    # ------------------------------------------------------------------
    # File transfer: shared uploads bucket + arbitrary workspace downloads
    # ------------------------------------------------------------------

    def _uploads_base(sid: str, agent) -> Path:
        """Where uploads for this session land: project root when the session
        is bound to a project, otherwise the agent's workspace."""
        proj = runtime.session_project(conn, sid)
        return Path(proj.root) if proj is not None else agent.workspace

    @app.post("/agents/{name}/sessions/{sid}/uploads")
    async def upload_file(name: str, sid: str, file: UploadFile = File(...)):
        agent = config.agents.get(name)
        if agent is None:
            raise HTTPException(404, "unknown agent")
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        if not file.filename:
            raise HTTPException(400, "missing filename")

        upload_base = _uploads_base(sid, agent)
        try:
            dest = ws.reserve_upload_filename(upload_base, file.filename)
        except ws.WorkspaceError as e:
            raise HTTPException(400, str(e))

        # Stream to disk, enforcing the cap.
        written = 0
        chunk_size = 64 * 1024
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                written += len(chunk)
                if written > UPLOAD_MAX_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, f"upload exceeds {UPLOAD_MAX_BYTES} bytes")
                out.write(chunk)

        # `dest` is inside upload_base (project root or workspace); express it
        # relative to that base for storage + wire response.
        rel = ws.relative_to_workspace(upload_base, dest)
        runtime.append_message(
            conn,
            sid,
            UploadMessage(path=rel, original_name=file.filename, size=written),
        )
        return {"path": rel, "size": written, "original_name": file.filename}

    @app.get("/agents/{name}/sessions/{sid}/uploads")
    def list_uploads(name: str, sid: str):
        agent = config.agents.get(name)
        if agent is None:
            raise HTTPException(404, "unknown agent")
        if not runtime.session_exists(conn, sid, name):
            raise HTTPException(404, "unknown session")
        uploads = ws.uploads_dir(_uploads_base(sid, agent))
        if not uploads.is_dir():
            return []
        entries = []
        for p in sorted(uploads.iterdir(), key=lambda e: e.stat().st_mtime, reverse=True):
            if p.is_file():
                stat = p.stat()
                entries.append(
                    {"path": f"{ws.UPLOADS_DIRNAME}/{p.name}", "size": stat.st_size, "mtime": int(stat.st_mtime)}
                )
        return entries

    def _workspace_or_404(name: str):
        agent = config.agents.get(name)
        if agent is None:
            raise HTTPException(404, "unknown agent")
        return agent

    def _workspace_resolve_or_400(agent, relative: str) -> Path:
        try:
            return ws.resolve(agent.workspace, relative)
        except ws.WorkspaceError as e:
            raise HTTPException(400, str(e))

    def _workspace_dir_listing(target: Path, relative: str) -> dict:
        entries = []
        for child in sorted(target.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
            st = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "is_dir": child.is_dir(),
                    "size": st.st_size if child.is_file() else 0,
                    "mtime": int(st.st_mtime * 1000),
                }
            )
        return {"path": relative, "entries": entries}

    @app.get("/agents/{name}/files")
    def list_workspace_root(name: str):
        agent = _workspace_or_404(name)
        agent.workspace.mkdir(parents=True, exist_ok=True)
        return _workspace_dir_listing(agent.workspace, "")

    @app.get("/agents/{name}/files/{path:path}")
    def get_workspace_path(name: str, path: str):
        agent = _workspace_or_404(name)
        full = _workspace_resolve_or_400(agent, path)
        if not full.exists():
            raise HTTPException(404, "not found")
        if full.is_dir():
            return _workspace_dir_listing(full, ws.relative_to_workspace(agent.workspace, full))
        return FileResponse(full, filename=full.name)

    @app.put("/agents/{name}/files/{path:path}")
    async def put_workspace_file(name: str, path: str, request: Request):
        agent = _workspace_or_404(name)
        full = _workspace_resolve_or_400(agent, path)
        full.parent.mkdir(parents=True, exist_ok=True)
        body = await request.body()
        full.write_bytes(body)
        return {
            "ok": True,
            "path": ws.relative_to_workspace(agent.workspace, full),
            "size": len(body),
        }

    @app.delete("/agents/{name}/files/{path:path}")
    def delete_workspace_path(name: str, path: str):
        agent = _workspace_or_404(name)
        full = _workspace_resolve_or_400(agent, path)
        # Defense in depth: never let a malformed path resolve to the
        # workspace root itself.
        if full == Path(agent.workspace).resolve():
            raise HTTPException(400, "cannot delete the workspace root")
        if not full.exists():
            raise HTTPException(404, "not found")
        if full.is_dir():
            import shutil

            shutil.rmtree(full)
        else:
            full.unlink()
        return {"ok": True}

    @app.post("/agents/{name}/files/{path:path}")
    def post_workspace_path(name: str, path: str, op: str = "", dest: str = ""):
        agent = _workspace_or_404(name)
        if op == "mkdir":
            full = _workspace_resolve_or_400(agent, path)
            full.mkdir(parents=True, exist_ok=True)
            return {"ok": True, "path": ws.relative_to_workspace(agent.workspace, full)}
        if op == "rename":
            if not dest:
                raise HTTPException(400, "?dest=<path> is required for op=rename")
            src = _workspace_resolve_or_400(agent, path)
            dst = _workspace_resolve_or_400(agent, dest)
            if not src.exists():
                raise HTTPException(404, "source not found")
            if dst.exists():
                raise HTTPException(409, "destination already exists")
            dst.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.move(str(src), str(dst))
            return {
                "ok": True,
                "from": ws.relative_to_workspace(agent.workspace, src),
                "to": ws.relative_to_workspace(agent.workspace, dst),
            }
        raise HTTPException(400, f"unsupported op: {op!r} (try ?op=mkdir or ?op=rename)")

    # ------------------------------------------------------------------
    # Unified per-client event stream + cross-session catch-up
    # ------------------------------------------------------------------

    @app.get("/events")
    def list_events(
        since_id: int | None = None,
        since_ms: int | None = None,
        limit: int = 200,
    ):
        """Catch-up query: persisted messages across every session, ordered by
        the monotonic message id. Use `since_id` for durable resume (the id
        the client last processed), or `since_ms` for a wall-clock-relative
        window. If neither is given, returns the most recent `limit` events."""

        limit = max(1, min(int(limit), 1000))
        sql = (
            "SELECT m.id, m.session_id, m.created_at, m.role, m.content_json, "
            "s.agent_name FROM messages m JOIN sessions s ON s.id = m.session_id"
        )
        where: list[str] = []
        params: list = []
        if since_id is not None:
            where.append("m.id > ?")
            params.append(int(since_id))
        if since_ms is not None:
            where.append("m.created_at >= ?")
            params.append(int(since_ms))
        if where:
            sql += " WHERE " + " AND ".join(where)
        if since_id is None and since_ms is None:
            # No cursor → return the *latest* `limit` rows. Order DESC, then
            # reverse client-side to keep ascending wire shape.
            sql += " ORDER BY m.id DESC LIMIT ?"
            params.append(limit + 1)
            rows = list(conn.execute(sql, params).fetchall())
            has_more = len(rows) > limit
            rows = list(reversed(rows[:limit]))
        else:
            sql += " ORDER BY m.id ASC LIMIT ?"
            params.append(limit + 1)
            rows = list(conn.execute(sql, params).fetchall())
            has_more = len(rows) > limit
            rows = rows[:limit]

        events: list[dict] = []
        for r in rows:
            msg = message_from_row(r["role"], json.loads(r["content_json"]))
            wire = _message_to_json(msg)
            events.append(
                {
                    "id": r["id"],
                    "session_id": r["session_id"],
                    "agent_name": r["agent_name"],
                    "created_at": r["created_at"],
                    **wire,
                }
            )
        return {
            "events": events,
            "next_since_id": events[-1]["id"] if events else since_id,
            "has_more": has_more,
        }

    @app.websocket("/events")
    async def ws_events(ws: WebSocket):
        """Single per-client WebSocket. Receives events for every session of
        every agent. Commands target a session via the `session_id` field."""

        token = ws.headers.get("authorization", "") or _qs_token(ws)
        if not _check_bearer(token, config.server.auth_secret):
            await ws.close(code=1008)
            return
        await ws.accept()
        queue = broker.subscribe_all()

        async def forwarder() -> None:
            try:
                while True:
                    event = await queue.get()
                    try:
                        await ws.send_json(event)
                    except Exception:  # noqa: BLE001
                        return
            except asyncio.CancelledError:
                return

        fwd_task = asyncio.create_task(forwarder())
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    cmd = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json(
                        {"type": "error", "code": "other", "message": "invalid JSON"}
                    )
                    continue
                t = cmd.get("type")
                sid = cmd.get("session_id")
                if t == "stop":
                    # v1: no-op (mid-turn cancellation isn't supported).
                    continue
                if t != "user_message":
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": "other",
                            "message": f"unsupported command type: {t!r}",
                        }
                    )
                    continue
                if not sid or not isinstance(sid, str):
                    await ws.send_json(
                        {
                            "type": "error",
                            "code": "other",
                            "message": "user_message requires a string `session_id`",
                        }
                    )
                    continue
                row = conn.execute(
                    "SELECT agent_name FROM sessions WHERE id = ?", (sid,)
                ).fetchone()
                if row is None:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": sid,
                            "code": "other",
                            "message": "unknown session",
                        }
                    )
                    continue
                agent = config.agents.get(row["agent_name"])
                if agent is None:
                    await ws.send_json(
                        {
                            "type": "error",
                            "session_id": sid,
                            "code": "other",
                            "message": f"unknown agent {row['agent_name']!r}",
                        }
                    )
                    continue
                # Spawn the turn as a background task — multiple sessions can
                # have turns running concurrently. Their events all flow back
                # through this same WS via the broker subscription.
                asyncio.create_task(
                    runtime.run_and_publish(
                        conn=conn,
                        config=config,
                        agent=agent,
                        session_id=sid,
                        user_text=cmd.get("text", ""),
                    )
                )
        except WebSocketDisconnect:
            return
        finally:
            fwd_task.cancel()
            broker.unsubscribe_all(queue)

    return app


def _check_bearer(header: str, expected: str) -> bool:
    if not header.lower().startswith("bearer "):
        return False
    return header[7:].strip() == expected


def _qs_token(ws: WebSocket) -> str:
    token = ws.query_params.get("token")
    return f"Bearer {token}" if token else ""


def _project_to_json(p) -> dict:
    return {
        "id": p.id,
        "name": p.name,
        "root": p.root,
        "description": p.description,
        "project_context": p.project_context,
        "created_at": p.created_at,
        "deleted_at": p.deleted_at,
    }


def _message_to_json(msg: Any) -> dict:
    if not is_dataclass(msg):
        return {"kind": type(msg).__name__, "data": {}}
    # An AssistantText with `injected_from` set represents a cross-session
    # injection (see post_to_session). Surface it under a distinct kind so
    # clients can render it consistently with the live `injected_message`
    # WS event, rather than seeing two different shapes for the same thing.
    if isinstance(msg, AssistantText) and msg.injected_from:
        return {
            "kind": "InjectedMessage",
            "data": {"text": msg.text, "from_session_id": msg.injected_from},
        }
    data = asdict(msg)
    # `thought_signature` is opaque bytes used internally to round-trip Gemini
    # thinking traces back to the model. Not meaningful to clients, and
    # FastAPI's default JSON encoder treats bytes as UTF-8 strings, which
    # explodes for binary content. Drop it before serializing.
    data.pop("thought_signature", None)
    return {"kind": type(msg).__name__, "data": data}
