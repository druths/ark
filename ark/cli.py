"""CLI client: `ark agents`, `ark sessions`, `ark chat`. Talks to the local
Ark server over REST + WebSocket.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import websockets

from . import config, paths


DEFAULT_DOWNLOAD_DIR = Path("ark-downloads")


def _client_settings() -> tuple[str, str]:
    """Return (base_url, auth_secret). Reads ~/.ark/config.json."""
    cfg = config.load()
    host = cfg.server.host
    # `0.0.0.0` / `::` are server bind addresses, not dial-able client targets.
    # In the containerized dev setup the server binds 0.0.0.0 and the host's
    # port-forward exposes 127.0.0.1:<port>.
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base_url = f"http://{host}:{cfg.server.port}"
    return base_url, cfg.server.auth_secret


def _headers(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------


def cmd_agents(_args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    r = httpx.get(f"{base_url}/agents", headers=_headers(secret), timeout=10)
    r.raise_for_status()
    for a in r.json():
        print(f"{a['name']:20s}  {a['provider']:12s}  {a['model']}")
    return 0


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------


def cmd_sessions(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    params: dict[str, Any] = {"limit": args.limit}
    if args.kind:
        params["kind"] = args.kind
    r = httpx.get(
        f"{base_url}/agents/{args.agent}/sessions",
        headers=_headers(secret),
        params=params,
        timeout=10,
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        print("(no sessions)")
        return 0
    for row in rows:
        ts = datetime.fromtimestamp(row["created_at"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{row['id']}  {row['kind']:14s}  {ts}")
    return 0


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


class _Ui:
    """Renders streaming assistant text plus a transient animated status line.

    Status (tool activity, thinking) lives on its own line below the cursor,
    drawn over with ANSI escape codes. The moment real assistant text starts
    streaming, the status line is wiped — and at end-of-turn it is wiped for
    good. The result: the transcript holds only the conversation, with a
    live indicator while the agent is working.
    """

    def __init__(self) -> None:
        self.is_tty = sys.stderr.isatty()
        self.status_text = ""
        self.status_visible = False
        self.text_pending_newline = False
        self.spinner_task: asyncio.Task | None = None

    def assistant_delta(self, text: str) -> None:
        self._wipe_status()
        sys.stdout.write(text)
        sys.stdout.flush()
        if text:
            self.text_pending_newline = not text.endswith("\n")

    def assistant_turn_end(self) -> None:
        if self.text_pending_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.text_pending_newline = False

    def status(self, text: str) -> None:
        self.status_text = text
        if not self.is_tty:
            sys.stderr.write(f"[{text}]\n")
            sys.stderr.flush()
            return
        if self.text_pending_newline:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.text_pending_newline = False
        self.status_visible = True
        self._render(_SPINNER[0])
        if self.spinner_task is None or self.spinner_task.done():
            self.spinner_task = asyncio.create_task(self._spinner_loop())

    def error_line(self, text: str) -> None:
        self._wipe_status()
        if self.text_pending_newline:
            sys.stdout.write("\n")
            self.text_pending_newline = False
        sys.stderr.write(f"\033[31m{text}\033[0m\n")
        sys.stderr.flush()

    def usage(
        self,
        input_tokens: int,
        output_tokens: int,
        context_window: int | None,
        model: str,
        cached_input_tokens: int = 0,
    ) -> None:
        """Dim per-turn token-usage indicator."""
        self._wipe_status()
        if self.text_pending_newline:
            sys.stdout.write("\n")
            self.text_pending_newline = False
        if context_window and context_window > 0:
            pct = input_tokens / context_window * 100
            msg = (
                f"{input_tokens:,}/{context_window:,} ctx ({pct:.1f}%)"
                f" · out {output_tokens:,}"
            )
        else:
            msg = f"in {input_tokens:,} · out {output_tokens:,}"
        if cached_input_tokens and input_tokens:
            msg += f" · {100 * cached_input_tokens / input_tokens:.0f}% cached"
        if model:
            msg += f" · {model}"
        sys.stderr.write(f"\033[2m[{msg}]\033[0m\n")
        sys.stderr.flush()

    def done(self) -> None:
        self._wipe_status()
        self.assistant_turn_end()

    def _wipe_status(self) -> None:
        if self.spinner_task and not self.spinner_task.done():
            self.spinner_task.cancel()
        self.spinner_task = None
        if self.status_visible and self.is_tty:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        self.status_visible = False

    def _render(self, glyph: str) -> None:
        if not self.is_tty:
            return
        sys.stderr.write(f"\r\033[K\033[2m{glyph} {self.status_text}\033[0m")
        sys.stderr.flush()

    async def _spinner_loop(self) -> None:
        i = 1
        try:
            while self.status_visible:
                await asyncio.sleep(0.1)
                if not self.status_visible:
                    break
                self._render(_SPINNER[i])
                i = (i + 1) % len(_SPINNER)
        except asyncio.CancelledError:
            pass


def _handle_event(ui: _Ui, evt: dict) -> None:
    t = evt.get("type")
    if t == "assistant_delta":
        ui.assistant_delta(evt.get("text", ""))
    elif t == "assistant_message":
        ui.assistant_turn_end()
    elif t == "thinking":
        ui.status("thinking")
    elif t == "tool_call":
        ui.status(f"running {evt.get('name', '?')}")
    elif t == "tool_result":
        if evt.get("error"):
            preview = (evt.get("output") or "").splitlines()[0][:80] if evt.get("output") else ""
            ui.error_line(f"[tool error: {preview}]" if preview else "[tool error]")
        ui.status("thinking")
    elif t == "turn_usage":
        ui.usage(
            int(evt.get("input_tokens", 0)),
            int(evt.get("output_tokens", 0)),
            evt.get("context_window"),
            evt.get("model") or "",
            cached_input_tokens=int(evt.get("cached_input_tokens", 0) or 0),
        )
    elif t == "error":
        code = evt.get("code") or "other"
        msg = evt.get("message") or ""
        if code == "context_too_long":
            ui.error_line(
                "[context too long — this session has exceeded the model's input "
                "window. Start a new session to continue.]"
            )
        elif code == "rate_limit":
            ui.error_line(f"[rate limited by provider — try again shortly. {msg[:120]}]")
        elif code == "auth":
            ui.error_line(f"[auth failure — check your provider API key. {msg[:120]}]")
        else:
            ui.error_line(f"[server error: {msg}]")
    elif t == "compaction_started":
        reason = evt.get("reason") or ""
        it = evt.get("input_tokens")
        cw = evt.get("context_window")
        pct = f" ({100 * it / cw:.0f}% full)" if it and cw else ""
        ui.status(f"compacting session{pct}")
        ui.error_line(f"[compaction started · reason={reason}]")
    elif t == "compaction_completed":
        summary = evt.get("summary") or ""
        ui.error_line(f"[compaction completed · summary {len(summary)} chars]")
    elif t == "compaction_failed":
        ui.error_line(
            f"[compaction failed: {evt.get('code','?')} — {(evt.get('message') or '')[:120]}]"
        )
    elif t == "compaction_skipped":
        ui.error_line(f"[compaction skipped: {evt.get('reason','')}]")
    elif t == "done":
        ui.done()


async def _upload_one(
    http: httpx.AsyncClient, agent: str, session_id: str, file_path: Path
) -> dict:
    if not file_path.is_file():
        raise FileNotFoundError(f"no such file: {file_path}")
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f, "application/octet-stream")}
        r = await http.post(f"/agents/{agent}/sessions/{session_id}/uploads", files=files)
    r.raise_for_status()
    return r.json()


async def _download_shared(
    http: httpx.AsyncClient, agent: str, path: str, dest_dir: Path
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(path).name
    target = dest_dir / name
    # auto-suffix to avoid overwriting prior downloads
    i = 2
    while target.exists():
        target = dest_dir / f"{Path(name).stem}-{i}{Path(name).suffix}"
        i += 1
    async with http.stream("GET", f"/agents/{agent}/files/{path}") as r:
        r.raise_for_status()
        with target.open("wb") as out:
            async for chunk in r.aiter_bytes(64 * 1024):
                out.write(chunk)
    return target


async def _chat(
    agent: str,
    session_id: str | None,
    history_only: bool = False,
    attachments: list[Path] | None = None,
    download_dir: Path | None = None,
    context: str | None = None,
) -> int:
    base_url, secret = _client_settings()
    headers = _headers(secret)
    dl_dir = download_dir or DEFAULT_DOWNLOAD_DIR

    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=120) as http:
        if session_id is None:
            body: dict = {}
            if context:
                body["context"] = context
            r = await http.post(f"/agents/{agent}/sessions", json=body)
            r.raise_for_status()
            session_id = r.json()["id"]
            print(f"[new session: {session_id}]", file=sys.stderr)
            if context:
                print("[context: seeded with --context]", file=sys.stderr)
        else:
            r = await http.get(f"/agents/{agent}/sessions/{session_id}/history")
            r.raise_for_status()
            for m in r.json():
                _print_history_entry(m)
            if history_only:
                return 0
            # If resuming and --context is passed, append it as a new context
            # message rather than seeding.
            if context:
                cr = await http.post(
                    f"/agents/{agent}/sessions/{session_id}/context",
                    json={"context": context},
                )
                cr.raise_for_status()
                print(
                    f"[context: appended (total {cr.json().get('count')})]",
                    file=sys.stderr,
                )

        # Pre-upload any --attach files before opening the WS so the agent
        # already sees them via list_uploads().
        for path in attachments or []:
            try:
                info = await _upload_one(http, agent, session_id, path)
                print(
                    f"[uploaded {path.name} → {info['path']} ({info['size']} bytes)]",
                    file=sys.stderr,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[upload failed for {path}: {e}]", file=sys.stderr)
                return 1

        ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        # Unified per-client endpoint: one WS, all events for all sessions.
        # We filter for the current session in the renderer; cross-session
        # activity (cron, other agents) flows past quietly.
        url = f"{ws_url}/events?token={secret}"
        async with websockets.connect(url) as ws:
            print(f"[connected — type your message, Ctrl-D to exit]", file=sys.stderr)
            ui = _Ui()
            stop = asyncio.Event()
            turn_done = asyncio.Event()
            turn_done.set()  # ready for first input

            async def reader():
                try:
                    async for raw in ws:
                        evt = json.loads(raw)
                        # Drop events that aren't for this session — this is a
                        # firehose now, so other sessions' activity comes too.
                        evt_sid = evt.get("session_id")
                        if evt_sid is not None and evt_sid != session_id:
                            continue
                        if evt.get("type") == "file_available":
                            await _on_file_available(http, agent, evt, dl_dir, ui)
                            continue
                        _handle_event(ui, evt)
                        if evt.get("type") == "done":
                            turn_done.set()
                except websockets.ConnectionClosed:
                    pass
                finally:
                    stop.set()
                    turn_done.set()  # unblock writer so it can exit

            async def writer():
                loop = asyncio.get_event_loop()
                while not stop.is_set():
                    await turn_done.wait()
                    if stop.is_set():
                        return
                    print("you> ", end="", flush=True, file=sys.stderr)
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    if line == "":  # EOF
                        await ws.close()
                        return
                    text = line.strip()
                    if not text:
                        continue
                    if text.startswith("/attach "):
                        path = Path(text[len("/attach "):].strip()).expanduser()
                        try:
                            info = await _upload_one(http, agent, session_id, path)
                            print(
                                f"[uploaded {path.name} → {info['path']} ({info['size']} bytes)]",
                                file=sys.stderr,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"[upload failed: {e}]", file=sys.stderr)
                        continue
                    if text.startswith("/context "):
                        ctx_text = text[len("/context "):].strip()
                        if not ctx_text:
                            print("[/context: empty, ignored]", file=sys.stderr)
                            continue
                        try:
                            cr = await http.post(
                                f"/agents/{agent}/sessions/{session_id}/context",
                                json={"context": ctx_text},
                            )
                            cr.raise_for_status()
                            print(
                                f"[context appended (total {cr.json().get('count')})]",
                                file=sys.stderr,
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"[/context failed: {e}]", file=sys.stderr)
                        continue
                    if text == "/compact" or text.startswith("/compact "):
                        # `/compact` → server-generated summary
                        # `/compact set: <text>` → client-supplied summary
                        rest = text[len("/compact"):].strip()
                        payload: dict = {}
                        if rest.startswith("set:"):
                            supplied = rest[len("set:"):].strip()
                            if not supplied:
                                print("[/compact set: empty, ignored]", file=sys.stderr)
                                continue
                            payload = {"summary": supplied}
                        elif rest:
                            print(
                                "[/compact: unrecognized form. Use `/compact` or "
                                "`/compact set: <your summary text>`]",
                                file=sys.stderr,
                            )
                            continue
                        try:
                            # Lifecycle events land on the WS naturally — the
                            # UI prints them via _handle_event. This just needs
                            # to be a fire-and-await (long timeout because the
                            # summarizer call is a full provider round-trip).
                            cr = await http.post(
                                f"/agents/{agent}/sessions/{session_id}/compact",
                                json=payload,
                                timeout=300,
                            )
                            if cr.status_code >= 400:
                                print(
                                    f"[/compact failed: {cr.status_code} {cr.text[:200]}]",
                                    file=sys.stderr,
                                )
                        except Exception as e:  # noqa: BLE001
                            print(f"[/compact failed: {e}]", file=sys.stderr)
                        continue
                    turn_done.clear()
                    ui.status("thinking")
                    await ws.send(
                        json.dumps(
                            {
                                "type": "user_message",
                                "session_id": session_id,
                                "text": text,
                            }
                        )
                    )

            await asyncio.gather(reader(), writer())
    return 0


async def _on_file_available(
    http: httpx.AsyncClient,
    agent: str,
    evt: dict,
    dl_dir: Path,
    ui: "_Ui",
) -> None:
    path = evt.get("path", "")
    size = evt.get("size", 0)
    desc = evt.get("description", "")
    try:
        local = await _download_shared(http, agent, path, dl_dir)
        suffix = f" — {desc}" if desc else ""
        ui.error_line(f"[agent shared {path} ({size} bytes) → {local}{suffix}]")
    except Exception as e:  # noqa: BLE001
        ui.error_line(f"[file_available download failed for {path}: {e}]")


def _print_history_entry(m: dict) -> None:
    kind = m.get("kind")
    data = m.get("data", {})
    if kind == "UserText":
        sys.stderr.write(f"\nyou> {data.get('text')}\n")
    elif kind == "AssistantText":
        sys.stdout.write(data.get("text", ""))
        sys.stdout.write("\n")
    elif kind == "ToolCall":
        sys.stderr.write(f"[tool {data.get('name')}(...)]\n")
    elif kind == "ToolResult":
        out = data.get("output", "")
        if len(out) > 400:
            out = out[:400] + "…"
        prefix = "tool error" if data.get("is_error") else "tool result"
        sys.stderr.write(f"[{prefix}: {out}]\n")


def cmd_chat(args: argparse.Namespace) -> int:
    attachments = [Path(p).expanduser() for p in (args.attach or [])]
    dl_dir = Path(args.download_dir).expanduser() if args.download_dir else None
    context = args.context
    if args.context_file:
        context = args.context_file.read()
    return asyncio.run(
        _chat(
            args.agent,
            args.session,
            attachments=attachments,
            download_dir=dl_dir,
            context=context,
        )
    )


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


def cmd_serve(_args: argparse.Namespace) -> int:
    import uvicorn

    from . import bootstrap, server

    if not paths.config_path().exists():
        print(f"no config at {paths.config_path()}. Run `python -m ark init` first.", file=sys.stderr)
        return 1
    cfg = config.load()
    bootstrap.bootstrap(cfg)  # idempotent — ensures agent dirs exist on every start
    app = server.create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")
    return 0


def cmd_heartbeat_set(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    body = {"interval_seconds": None if args.seconds in (0, "0") else int(args.seconds)}
    r = httpx.put(
        f"{base_url}/agents/{args.agent}/heartbeat",
        headers=_headers(secret),
        json=body,
        timeout=10,
    )
    r.raise_for_status()
    print(f"heartbeat: {'disabled' if body['interval_seconds'] is None else str(body['interval_seconds']) + 's'}")
    return 0


def cmd_cron_list(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    r = httpx.get(
        f"{base_url}/agents/{args.agent}/crons", headers=_headers(secret), timeout=10
    )
    r.raise_for_status()
    rows = r.json()
    if not rows:
        print("(no crons)")
        return 0
    for c in rows:
        status = "" if c["enabled"] else " [disabled]"
        # Project attribution (server returns project_id + project_name).
        proj = ""
        if c.get("project_id"):
            pname = c.get("project_name")
            proj = f" [project={pname}]" if pname else f" [project={c['project_id']} DELETED]"
        prompt = c["prompt"]
        if args.full:
            # Indent multi-line prompts under their header.
            indented = "\n".join("    " + line for line in prompt.splitlines())
            print(f"{c['id']}: {c['expr']}{status}{proj}")
            print(indented)
        else:
            # One-line preview. Collapse newlines so we don't break the table.
            preview = " ".join(prompt.split())
            if len(preview) > 100:
                preview = preview[:100] + "…"
            print(f"{c['id']}: {c['expr']}{status}{proj} — {preview}")
    return 0


def cmd_cron_set(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    prompt = args.prompt_file.read_text() if args.prompt_file else args.prompt
    if not prompt:
        print("--prompt or --prompt-file is required", file=sys.stderr)
        return 1
    headers = _headers(secret)
    body: dict[str, Any] = {"expr": args.expr, "prompt": prompt}
    # --project resolves a name to id; --no-project explicitly detaches.
    # Neither flag → server keeps whatever project_id was already set.
    if args.no_project:
        body["project_id"] = None
    elif args.project is not None:
        pl = httpx.get(f"{base_url}/projects", headers=headers, timeout=10)
        pl.raise_for_status()
        matches = [p for p in pl.json() if p["name"] == args.project]
        if not matches:
            print(f"error: no project named '{args.project}'", file=sys.stderr)
            return 2
        body["project_id"] = matches[0]["id"]
    r = httpx.put(
        f"{base_url}/agents/{args.agent}/crons/{args.id}",
        headers=headers,
        json=body,
        timeout=10,
    )
    if r.status_code >= 400:
        print(f"error: {r.text}", file=sys.stderr)
        return 1
    print(f"cron '{args.id}' saved")
    return 0


def cmd_cron_remove(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    r = httpx.delete(
        f"{base_url}/agents/{args.agent}/crons/{args.id}",
        headers=_headers(secret),
        timeout=10,
    )
    if r.status_code == 404:
        print("not found", file=sys.stderr)
        return 1
    r.raise_for_status()
    print(f"cron '{args.id}' removed")
    return 0


def cmd_cron_history(args: argparse.Namespace) -> int:
    base_url, secret = _client_settings()
    r = httpx.get(
        f"{base_url}/agents/{args.agent}/crons/{args.id}/sessions",
        headers=_headers(secret),
        params={"limit": args.limit},
        timeout=10,
    )
    if r.status_code == 404:
        print(f"unknown agent: {args.agent}", file=sys.stderr)
        return 1
    r.raise_for_status()
    rows = r.json()
    if not rows:
        print(f"(no fires recorded for cron '{args.id}')")
        return 0
    print(f"{'SESSION':38s}  {'FIRED':19s}  {'STATUS':7s}  SUMMARY")
    for row in rows:
        sid = row["session_id"]
        ts = datetime.fromtimestamp(row["created_at"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        if row.get("had_error"):
            status = "error"
        elif row.get("summary", "(no output)") == "(no output)":
            status = "empty"
        else:
            status = "ok"
        # Summary printed as a single line — collapse newlines, truncate.
        summary = " ".join((row.get("summary") or "").split())
        if status == "error":
            summary = f"{row.get('error_code') or 'other'}: {summary}".strip(": ")
        if len(summary) > 80:
            summary = summary[:80] + "…"
        print(f"{sid:38s}  {ts}  {status:7s}  {summary}")
    return 0


def cmd_session_set_project(args: argparse.Namespace) -> int:
    """Reassign (or detach) a session's project binding."""
    base_url, secret = _client_settings()
    sid = args.session_id
    headers = _headers(secret)

    # Resolve agent from session metadata (avoids requiring --agent).
    meta_r = httpx.get(
        f"{base_url}/sessions/{sid}", headers=headers, timeout=10
    )
    if meta_r.status_code == 404:
        print(f"error: session {sid} not found", file=sys.stderr)
        return 2
    meta_r.raise_for_status()
    agent = meta_r.json()["agent_name"]

    project_id: str | None
    if args.none:
        project_id = None
    else:
        # Accept a project name; look up its id via GET /projects.
        pl = httpx.get(f"{base_url}/projects", headers=headers, timeout=10)
        pl.raise_for_status()
        matches = [p for p in pl.json() if p["name"] == args.project]
        if not matches:
            print(f"error: no project named '{args.project}'", file=sys.stderr)
            return 2
        project_id = matches[0]["id"]

    r = httpx.patch(
        f"{base_url}/agents/{agent}/sessions/{sid}/project",
        headers=headers,
        json={"project_id": project_id},
        timeout=15,
    )
    if r.status_code != 200:
        print(f"error: {r.status_code} {r.text}", file=sys.stderr)
        return 2
    result = r.json()
    if not result.get("changed"):
        print("no change — session is already assigned as requested")
        return 0

    def _label(p):
        if p is None:
            return "(no project)"
        return f"{p['name']} [{p['id']}]"

    print(
        f"reassigned session {sid}: "
        f"{_label(result.get('from'))} → {_label(result.get('to'))}"
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Pretty-print a session transcript. Works for any session kind."""
    base_url, secret = _client_settings()
    headers = _headers(secret)
    # Metadata first
    r = httpx.get(f"{base_url}/sessions/{args.session_id}", headers=headers, timeout=10)
    if r.status_code == 404:
        print(f"unknown session: {args.session_id}", file=sys.stderr)
        return 1
    r.raise_for_status()
    meta = r.json()
    # Full history
    r = httpx.get(
        f"{base_url}/agents/{meta['agent_name']}/sessions/{args.session_id}/history",
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    history = r.json()
    _format_session(meta, history)
    return 0


def _format_session(meta: dict, history: list[dict]) -> None:
    kind = meta["kind"]
    sid = meta["id"]
    created = datetime.fromtimestamp(meta["created_at"] / 1000).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"=== {kind} session {sid} ({meta['agent_name']}, fired {created}) ==="
    print(header)
    if meta.get("cron_id"):
        print(f"cron: {meta['cron_id']}")
    if meta.get("cron_prompt"):
        prompt_preview = meta["cron_prompt"]
        if len(prompt_preview) > 200:
            prompt_preview = prompt_preview[:200] + "…"
        print(f"prompt: {prompt_preview!r}")
    if meta.get("project_id"):
        print(f"project: {meta['project_id']}")
    print()
    turn_index = 0
    for m in history:
        k = m["kind"]
        d = m.get("data", {})
        if k == "UserText":
            turn_index += 1
            print(f"[turn {turn_index}] user> {_truncate(d.get('text', ''), 200)}")
        elif k == "AssistantText":
            text = d.get("text") or ""
            if text:
                print(f"  → {_truncate(text, 200)}")
        elif k == "InjectedMessage":
            print(f"  ← injected from {d.get('from_session_id', '')[:8]}…: {_truncate(d.get('text', ''), 120)}")
        elif k == "ToolCall":
            name = d.get("name", "?")
            inp = d.get("input") or {}
            arg_summary = ", ".join(f"{k}=…" for k in list(inp.keys())[:3])
            print(f"  → tool: {name}({arg_summary})")
        elif k == "ToolResult":
            err = " ✗" if d.get("is_error") else ""
            out = (d.get("output") or "").strip()
            print(f"     ↳{err} {_truncate(out, 120)}")
        elif k == "UploadMessage":
            print(f"  ← upload: {d.get('original_name')} → {d.get('path')} ({d.get('size')} bytes)")
        elif k == "SharedFile":
            desc = f" — {d['description']}" if d.get("description") else ""
            print(f"  → shared: {d.get('path')} ({d.get('size')} bytes){desc}")
        elif k == "TurnMetrics":
            print(f"     [metrics: in={d.get('input_tokens')} out={d.get('output_tokens')}]")
        elif k == "RunError":
            print(f"  ✗ error: {d.get('code')} — {_truncate(d.get('message', ''), 200)}")
        elif k == "SessionContext":
            print(f"  [context: {_truncate(d.get('text', ''), 200)}]")
        else:
            print(f"  [{k}]")


def _truncate(s: str, n: int) -> str:
    s = " ".join(s.split())  # collapse whitespace
    if len(s) <= n:
        return s
    return s[:n] + "…"


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def cmd_init(_args: argparse.Namespace) -> int:
    from . import bootstrap, db

    if not paths.config_path().exists():
        print(
            f"no config at {paths.config_path()}. Create it before running init.",
            file=sys.stderr,
        )
        return 1
    cfg = config.load()
    bootstrap.bootstrap(cfg)
    conn = db.init_db()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    print(f"ark home: {paths.ark_home()}")
    print(f"agents:   {', '.join(cfg.agents) or '(none configured)'}")
    print(f"db:       {paths.db_path()} (schema v{version})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ark")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="bootstrap ARK_HOME and apply DB migrations").set_defaults(
        func=cmd_init
    )

    sub.add_parser("agents", help="list configured agents").set_defaults(func=cmd_agents)

    sess = sub.add_parser("sessions", help="list an agent's sessions")
    sess.add_argument("agent")
    sess.add_argument("--kind", help="filter by session kind")
    sess.add_argument("--limit", type=int, default=50)
    sess.set_defaults(func=cmd_sessions)

    chat = sub.add_parser("chat", help="open an interactive session")
    chat.add_argument("agent")
    chat.add_argument("--session", help="resume an existing session id")
    chat.add_argument(
        "--attach",
        action="append",
        metavar="PATH",
        help="upload a file before chatting (repeatable). The agent sees uploads in its uploads/ dir.",
    )
    chat.add_argument(
        "--download-dir",
        metavar="DIR",
        help=f"where to save files the agent shares (default: ./{DEFAULT_DOWNLOAD_DIR})",
    )
    chat.add_argument(
        "--context",
        metavar="TEXT",
        help="additional instructions for the agent in this session (appended to system prompt)",
    )
    chat.add_argument(
        "--context-file",
        type=argparse.FileType("r"),
        metavar="PATH",
        help="read --context from a file instead of inline",
    )
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="run the Ark server")
    serve.set_defaults(func=cmd_serve)

    hb = sub.add_parser("heartbeat", help="set or disable an agent's heartbeat")
    hb.add_argument("agent")
    hb.add_argument("seconds", help="interval in seconds (0 to disable)")
    hb.set_defaults(func=cmd_heartbeat_set)

    cron = sub.add_parser("cron", help="manage an agent's cron entries")
    cron_sub = cron.add_subparsers(dest="cron_command", required=True)

    cron_list = cron_sub.add_parser("list", help="list crons for an agent")
    cron_list.add_argument("agent")
    cron_list.add_argument(
        "--full",
        "-l",
        action="store_true",
        help="print full multi-line prompts (default: a 100-char single-line preview)",
    )
    cron_list.set_defaults(func=cmd_cron_list)

    cron_set = cron_sub.add_parser("set", help="add or update a cron entry")
    cron_set.add_argument("agent")
    cron_set.add_argument("id")
    cron_set.add_argument("expr", help="5-field UNIX cron expression")
    cron_set.add_argument("--prompt", help="inline prompt body")
    cron_set.add_argument(
        "--prompt-file", type=argparse.FileType("r"), help="read prompt from file"
    )
    proj_grp = cron_set.add_mutually_exclusive_group()
    proj_grp.add_argument(
        "--project",
        help="bind each fire's session to this project (by name)",
    )
    proj_grp.add_argument(
        "--no-project",
        action="store_true",
        help="explicitly detach any project binding on the cron",
    )
    cron_set.set_defaults(func=cmd_cron_set)

    cron_remove = cron_sub.add_parser("remove", help="remove a cron entry")
    cron_remove.add_argument("agent")
    cron_remove.add_argument("id")
    cron_remove.set_defaults(func=cmd_cron_remove)

    cron_history = cron_sub.add_parser(
        "history", help="show recent fires of a specific cron entry"
    )
    cron_history.add_argument("agent")
    cron_history.add_argument("id")
    cron_history.add_argument("--limit", type=int, default=20)
    cron_history.set_defaults(func=cmd_cron_history)

    show = sub.add_parser(
        "show", help="pretty-print a session's transcript (any session kind)"
    )
    show.add_argument("session_id")
    show.set_defaults(func=cmd_show)

    session = sub.add_parser("session", help="manage a session")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    set_project = session_sub.add_parser(
        "set-project",
        help="reassign or detach a session's project binding",
    )
    set_project.add_argument("session_id")
    grp = set_project.add_mutually_exclusive_group(required=True)
    grp.add_argument("project", nargs="?", help="project name to assign")
    grp.add_argument(
        "--none", action="store_true", help="detach the session from any project"
    )
    set_project.set_defaults(func=cmd_session_set_project)

    return parser
