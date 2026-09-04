"""Heartbeat + cron scheduler.

A long-running asyncio task that wakes once per second, decides which timers
should fire, and starts a fresh session for each. Heartbeat intervals come
from `agent_state.heartbeat_seconds`; cron entries come from the `crons`
table (managed by the schedule meta-tool).

State is in-memory: `last_fired` per agent / per cron entry. On startup, all
`last_fired` values are seeded to "now" so timers don't fire immediately.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
from datetime import datetime, timezone

from croniter import croniter

from . import paths, runtime
from .config import Config


class Scheduler:
    def __init__(self, conn: sqlite3.Connection, config: Config) -> None:
        self.conn = conn
        self.config = config
        self.last_heartbeat: dict[str, float] = {}
        self.last_cron: dict[tuple[str, str], float] = {}
        self._task: asyncio.Task | None = None
        # asyncio.Event() needs a running loop on Python 3.9 — defer until start().
        self._stop: asyncio.Event | None = None

    def start(self) -> None:
        if self._task is None:
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        now = time.time()
        for name in self.config.agents:
            self.last_heartbeat[name] = now
        for row in self.conn.execute(
            "SELECT agent_name, id FROM crons WHERE enabled = 1"
        ).fetchall():
            self.last_cron[(row["agent_name"], row["id"])] = now

        while not self._stop.is_set():
            try:
                await self._tick()
            except Exception as e:  # noqa: BLE001
                print(f"[scheduler] tick error: {e}", file=sys.stderr)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

    async def _tick(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        # heartbeats
        for name, agent in self.config.agents.items():
            interval = self._heartbeat_interval(name)
            if not interval:
                continue
            # `.setdefault` (not `.get`) so newly-configured agents anchor
            # their "last fired" at first observation rather than being reset
            # to now on every tick.
            last = self.last_heartbeat.setdefault(name, now)
            if now - last >= interval:
                self.last_heartbeat[name] = now
                asyncio.create_task(self._fire_heartbeat(name))

        # crons
        rows = self.conn.execute(
            "SELECT agent_name, id, expr, prompt, project_id, max_tokens "
            "FROM crons WHERE enabled = 1"
        ).fetchall()
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        for row in rows:
            key = (row["agent_name"], row["id"])
            # Same `.setdefault` pattern — without this, crons added after
            # startup would never fire because last_cron[key] would never get
            # written.
            last = self.last_cron.setdefault(key, now)
            last_dt = datetime.fromtimestamp(last, tz=timezone.utc)
            try:
                next_dt = croniter(row["expr"], last_dt).get_next(datetime)
            except Exception as e:  # noqa: BLE001
                print(f"[scheduler] cron {row['id']!r}: {e}", file=sys.stderr)
                continue
            if now_dt >= next_dt:
                self.last_cron[key] = now
                asyncio.create_task(
                    self._fire_cron(
                        row["agent_name"], row["id"], row["prompt"],
                        row["project_id"], row["max_tokens"],
                    )
                )

    def _heartbeat_interval(self, agent_name: str) -> int | None:
        row = self.conn.execute(
            "SELECT heartbeat_seconds FROM agent_state WHERE agent_name = ?",
            (agent_name,),
        ).fetchone()
        if row is None:
            return None
        return row["heartbeat_seconds"]

    async def _fire_heartbeat(self, agent_name: str) -> None:
        agent = self.config.agents.get(agent_name)
        if agent is None:
            return
        prompt_path = paths.agent_dir(agent_name) / "heartbeat_prompt.md"
        prompt = (
            prompt_path.read_text()
            if prompt_path.exists()
            else "This is your scheduled heartbeat. Check on anything that needs attention."
        )
        await self._drive(agent_name, "heartbeat", prompt)

    async def _fire_cron(
        self, agent_name: str, cron_id: str, prompt: str,
        project_id: str | None, max_tokens: int | None = None,
    ) -> None:
        # If the cron was bound to a project that has since been soft-deleted,
        # warn but still fire — the session just runs project-less. Matches
        # the "runtime.session_project returns None for deleted" semantics.
        if project_id is not None:
            from . import projects as _projects

            p = _projects.get(self.conn, project_id)
            if p is None or p.deleted_at is not None:
                print(
                    f"[scheduler] cron {cron_id!r} for {agent_name}: bound project "
                    f"{project_id} is deleted, firing in workspace mode",
                    file=sys.stderr,
                )
                # We still pass the id through — the session row records what
                # the cron was TRYING to bind to (audit). runtime.session_project
                # returns None so the LLM view is project-less.
        await self._drive(
            agent_name, "cron", prompt,
            cron_id=cron_id, project_id=project_id, max_tokens=max_tokens,
        )

    async def _drive(
        self,
        agent_name: str,
        kind: str,
        prompt: str,
        cron_id: str | None = None,
        project_id: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        agent = self.config.agents.get(agent_name)
        if agent is None:
            return
        sid = runtime.create_session(
            self.conn, agent_name, kind=kind, cron_id=cron_id, project_id=project_id
        )
        try:
            # run_and_publish routes events through the broker so connected
            # clients on the unified /events WS see scheduled-session activity
            # just like they see user-driven turns.
            await runtime.run_and_publish(
                conn=self.conn,
                config=self.config,
                agent=agent,
                session_id=sid,
                user_text=prompt,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[scheduler] {kind} session {sid} error: {e}", file=sys.stderr)
