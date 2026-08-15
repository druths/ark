"""SQLite connection + schema migrations.

Schema version is tracked via SQLite's `user_version` PRAGMA. Each migration is
a tuple (target_version, sql). To add a new migration, append a tuple — never
edit a past one.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import paths

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE sessions (
          id TEXT PRIMARY KEY,
          agent_name TEXT NOT NULL,
          kind TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          ended_at INTEGER
        );

        CREATE INDEX idx_sessions_agent ON sessions(agent_name, created_at DESC);

        CREATE TABLE messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          seq INTEGER NOT NULL,
          role TEXT NOT NULL,
          content_json TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          UNIQUE (session_id, seq)
        );

        CREATE TABLE agent_state (
          agent_name TEXT PRIMARY KEY,
          heartbeat_seconds INTEGER
        );

        CREATE TABLE crons (
          agent_name TEXT NOT NULL,
          id TEXT NOT NULL,
          expr TEXT NOT NULL,
          prompt TEXT NOT NULL,
          enabled INTEGER NOT NULL DEFAULT 1,
          PRIMARY KEY (agent_name, id)
        );
        """,
    ),
    (
        2,
        # Index for the `/events?since_ms=...` catch-up query (cross-session
        # scan ordered by wall-clock time). The id-based catch-up uses the
        # primary key, so it doesn't need a separate index.
        "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);",
    ),
    (
        3,
        """
        CREATE TABLE projects (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          root TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          project_context TEXT NOT NULL DEFAULT '',
          created_at INTEGER NOT NULL,
          deleted_at INTEGER
        );

        -- Names must be unique among active (non-deleted) projects, but a
        -- soft-deleted project's name can be reused for a fresh one.
        CREATE UNIQUE INDEX idx_projects_name_active
          ON projects(name) WHERE deleted_at IS NULL;

        ALTER TABLE sessions ADD COLUMN project_id TEXT REFERENCES projects(id);
        CREATE INDEX idx_sessions_project ON sessions(project_id);
        """,
    ),
    (
        4,
        # Track which cron entry triggered each cron-kind session, so we can
        # query "show me the fires of cron X" without trying to infer from
        # the prompt text. Sessions that aren't cron fires keep this NULL.
        # No FK to crons: sessions outlive cron edits/deletions and the
        # cron_id label is meant to be stable even if the cron is later removed.
        """
        ALTER TABLE sessions ADD COLUMN cron_id TEXT;
        CREATE INDEX idx_sessions_cron ON sessions(agent_name, cron_id, created_at DESC);
        """,
    ),
    (
        5,
        # Cron entries can be bound to a project, so their fires spawn sessions
        # already attached to that project (system prompt gets the project
        # stanza from turn 1, uploads land in the project's uploads dir, etc.).
        # Nullable — existing crons keep firing project-less sessions unchanged.
        # No FK: we intentionally allow a cron to reference a soft-deleted
        # project without cascading. The scheduler handles that gracefully.
        "ALTER TABLE crons ADD COLUMN project_id TEXT;",
    ),
    (
        6,
        # Per-cron override of the turn's token budget. When null, resolution
        # falls back to AgentConfig.max_turn_tokens, then to the runtime's
        # DEFAULT_TURN_TOKEN_BUDGET constant. Replaces the old hardcoded
        # max_iterations=16 cap on the model→tools loop.
        "ALTER TABLE crons ADD COLUMN max_tokens INTEGER;",
    ),
    (
        7,
        # Opaque client-supplied session metadata (JSON object). Stored
        # server-side only and surfaced to skills via ToolContext.metadata —
        # the same unforgeable channel as session_id. NEVER rendered into
        # the system prompt or model-visible context: it exists precisely to
        # carry per-session capabilities (e.g. a callback URL + secret pair)
        # that must not transit the model.
        "ALTER TABLE sessions ADD COLUMN metadata_json TEXT;",
    ),
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or paths.db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: FastAPI dispatches sync route handlers to a thread
    # pool, so a single shared connection is read/written from multiple threads.
    # SQLite serializes its own writes; WAL mode handles concurrent readers.
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, sql in MIGRATIONS:
        if target <= current:
            continue
        conn.executescript("BEGIN;\n" + sql + f"\nPRAGMA user_version = {target};\nCOMMIT;")
        current = target
    return current


def init_db(path: Path | None = None) -> sqlite3.Connection:
    conn = connect(path)
    migrate(conn)
    return conn
