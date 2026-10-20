"""End-to-end: upload + download + share_with_client + provider folding."""

import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from ark import broker, db, runtime, tools, workspace as ws
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.provider import to_anthropic_messages, to_openai_messages
from ark.server import create_app
from ark.tools import ToolContext
from ark.types import SharedFile, UploadMessage


def make_config(workspace):
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"anthropic": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe",
                provider="anthropic",
                model="m",
                workspace=workspace,
            )
        },
    )


# ---------------------------------------------------------------------------
# REST: upload + list + download
# ---------------------------------------------------------------------------


def test_rest_upload_list_download(ark_home, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = make_config(workspace)
    app = create_app(cfg)
    client = TestClient(app)
    headers = {"Authorization": "Bearer x"}

    # Create a session.
    r = client.post("/agents/scribe/sessions", headers=headers)
    assert r.status_code == 200
    sid = r.json()["id"]

    # Upload a file.
    r = client.post(
        f"/agents/scribe/sessions/{sid}/uploads",
        headers=headers,
        files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == "uploads/hello.txt"
    assert body["size"] == 11
    assert (workspace / "uploads" / "hello.txt").read_bytes() == b"hello world"

    # Re-upload same name → auto-suffixed.
    r = client.post(
        f"/agents/scribe/sessions/{sid}/uploads",
        headers=headers,
        files={"file": ("hello.txt", io.BytesIO(b"second"), "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["path"] == "uploads/hello-2.txt"

    # List shows both, newest first.
    r = client.get(f"/agents/scribe/sessions/{sid}/uploads", headers=headers)
    assert r.status_code == 200
    names = [e["path"] for e in r.json()]
    assert "uploads/hello.txt" in names
    assert "uploads/hello-2.txt" in names

    # Download the original.
    r = client.get("/agents/scribe/files/uploads/hello.txt", headers=headers)
    assert r.status_code == 200
    assert r.content == b"hello world"

    # Upload was persisted as an UploadMessage in session history.
    history = runtime.load_history(app.state.conn, sid)
    upload_msgs = [m for m in history if isinstance(m, UploadMessage)]
    assert len(upload_msgs) == 2
    assert upload_msgs[0].original_name == "hello.txt"
    assert upload_msgs[0].size == 11


def test_rest_upload_size_cap(ark_home, tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = make_config(workspace)
    # Shrink the cap so we don't have to ship 25MB through the test client.
    import ark.server as srv

    monkeypatch.setattr(srv, "UPLOAD_MAX_BYTES", 10)
    app = create_app(cfg)
    client = TestClient(app)
    headers = {"Authorization": "Bearer x"}
    r = client.post("/agents/scribe/sessions", headers=headers)
    sid = r.json()["id"]
    r = client.post(
        f"/agents/scribe/sessions/{sid}/uploads",
        headers=headers,
        files={"file": ("big.bin", io.BytesIO(b"x" * 100), "application/octet-stream")},
    )
    assert r.status_code == 413
    assert not (workspace / "uploads" / "big.bin").exists()


def test_rest_download_never_leaks_external_file(ark_home, tmp_path):
    """Strong invariant: regardless of the URL encoding tried, an attacker
    must not be able to read a file outside the workspace via the download
    endpoint. ws.resolve() is the actual guard — this test exists so any
    future refactor that moves or skips it fails loudly.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret_content = "SHOULD-NEVER-LEAK-OUTSIDE-WORKSPACE"
    secret.write_text(secret_content)

    app = create_app(make_config(workspace))
    client = TestClient(app)
    headers = {"Authorization": "Bearer x"}

    attempts = [
        "../secret.txt",
        "..%2Fsecret.txt",
        "%2e%2e%2Fsecret.txt",
        "a/..%2F..%2Fsecret.txt",
        "subdir/../../secret.txt",
        "/" + str(secret),  # absolute path attempt
    ]
    for path in attempts:
        r = client.get(f"/agents/scribe/files/{path}", headers=headers)
        assert r.status_code in (400, 404), (path, r.status_code)
        assert secret_content not in r.text, f"leaked via {path!r}: {r.text}"


def test_rest_download_symlink_escape_blocked(ark_home, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("EXTERNAL")
    (workspace / "trap").symlink_to(outside)
    app = create_app(make_config(workspace))
    client = TestClient(app)
    headers = {"Authorization": "Bearer x"}
    r = client.get("/agents/scribe/files/trap", headers=headers)
    assert r.status_code == 400
    assert "EXTERNAL" not in r.text


# ---------------------------------------------------------------------------
# Agent tools: list_uploads + share_with_client
# ---------------------------------------------------------------------------


def make_ctx(conn, workspace, agent_name="scribe", session_id="s"):
    agent = AgentConfig(
        name=agent_name,
        provider="anthropic",
        model="m",
        workspace=workspace,
    )
    return ToolContext(
        conn=conn,
        config=make_config(workspace),
        agent=agent,
        session_id=session_id,
        cwd=workspace,
        loaded_skills=set(),
    )


def test_tool_list_uploads(ark_home, tmp_path):
    conn = db.init_db()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = make_ctx(conn, workspace)
    output, err = asyncio.run(tools.execute("list_uploads", {}, ctx=ctx))
    assert err is False
    assert "(no uploads)" in output

    ws.ensure_uploads_dir(workspace)
    (workspace / "uploads" / "a.txt").write_text("hi")
    output, err = asyncio.run(tools.execute("list_uploads", {}, ctx=ctx))
    assert err is False
    assert "uploads/a.txt" in output


@pytest.mark.asyncio
async def test_tool_share_with_client_persists_and_publishes(ark_home, tmp_path):
    conn = db.init_db()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sid = runtime.create_session(conn, "scribe", "conversational")
    ctx = make_ctx(conn, workspace, session_id=sid)

    chart = workspace / "chart.png"
    chart.write_bytes(b"\x89PNG fake")

    queue = broker.subscribe(sid)
    try:
        output, err = await tools.execute(
            "share_with_client",
            {"path": "chart.png", "description": "Q4"},
            ctx=ctx,
        )
        assert err is False, output

        # Persisted as SharedFile.
        history = runtime.load_history(conn, sid)
        shared = [m for m in history if isinstance(m, SharedFile)]
        assert len(shared) == 1
        assert shared[0].path == "chart.png"
        assert shared[0].description == "Q4"
        assert shared[0].size == len(b"\x89PNG fake")

        # Broker delivered file_available.
        evt = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert evt["type"] == "file_available"
        assert evt["path"] == "chart.png"
        assert evt["description"] == "Q4"
        # session_id + agent_name must be tagged so the unified /events WS
        # firehose can route this to clients.
        assert evt["session_id"] == sid
        assert evt["agent_name"] == "scribe"
    finally:
        broker.unsubscribe(sid, queue)


def test_tool_share_with_client_rejects_outside_workspace(ark_home, tmp_path):
    conn = db.init_db()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sid = runtime.create_session(conn, "scribe", "conversational")
    ctx = make_ctx(conn, workspace, session_id=sid)
    output, err = asyncio.run(
        tools.execute("share_with_client", {"path": "../escape.txt"}, ctx=ctx)
    )
    assert err is True
    assert "escapes" in output.lower() or "not a file" in output.lower()


def test_tool_share_with_client_rejects_missing(ark_home, tmp_path):
    conn = db.init_db()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sid = runtime.create_session(conn, "scribe", "conversational")
    ctx = make_ctx(conn, workspace, session_id=sid)
    output, err = asyncio.run(
        tools.execute("share_with_client", {"path": "nope.png"}, ctx=ctx)
    )
    assert err is True
    assert "not a file" in output.lower()


# ---------------------------------------------------------------------------
# Dual-root share: workspace + project root both acceptable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_share_absolute_path_under_workspace(ark_home, tmp_path):
    """An absolute path pointing INTO the agent's workspace resolves as
    `root: "workspace"` — the share goes through the same download endpoint
    as today's relative-path form."""
    from ark import projects as _projects

    conn = db.init_db()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sid = runtime.create_session(conn, "scribe", "conversational")
    ctx = make_ctx(conn, workspace, session_id=sid)

    chart = workspace / "chart.png"
    chart.write_bytes(b"\x89PNG fake")

    output, err = await tools.execute(
        "share_with_client",
        {"path": str(chart.resolve()), "description": "Q4"},
        ctx=ctx,
    )
    assert err is False, output
    history = runtime.load_history(conn, sid)
    sf = [m for m in history if isinstance(m, SharedFile)][0]
    assert sf.root == "workspace"
    assert sf.project_id is None
    assert sf.path == "chart.png"


@pytest.mark.asyncio
async def test_share_absolute_path_in_project_root(ark_home, tmp_path):
    """An absolute path under the session's bound project root is accepted
    without needing to copy into workspace first. Row carries
    `root: "project"` + project_id so clients fetch via the project files
    endpoint rather than the agent files endpoint."""
    from ark import projects as _projects

    conn = db.init_db()
    workspace = tmp_path / "ws"; workspace.mkdir()
    project_root = tmp_path / "proj"; project_root.mkdir()
    p = _projects.create(
        conn, name="demo", root=str(project_root),
        description="", project_context="",
    )
    sid = runtime.create_session(conn, "scribe", "conversational", project_id=p.id)
    ctx = make_ctx(conn, workspace, session_id=sid)

    report = project_root / "Q3-report.pdf"
    report.write_bytes(b"%PDF-1.4\n...")

    queue = broker.subscribe(sid)
    try:
        output, err = await tools.execute(
            "share_with_client",
            {"path": str(report.resolve()), "description": "quarterly"},
            ctx=ctx,
        )
        assert err is False, output
        # Row records the project binding.
        history = runtime.load_history(conn, sid)
        sf = [m for m in history if isinstance(m, SharedFile)][0]
        assert sf.root == "project"
        assert sf.project_id == p.id
        assert sf.path == "Q3-report.pdf"
        assert sf.size == len(b"%PDF-1.4\n...")

        # Broker frame carries root + project_id so the client knows which
        # endpoint to fetch from.
        evt = await asyncio.wait_for(queue.get(), timeout=0.5)
        assert evt["type"] == "file_available"
        assert evt["root"] == "project"
        assert evt["project_id"] == p.id
        assert evt["path"] == "Q3-report.pdf"
    finally:
        broker.unsubscribe(sid, queue)


@pytest.mark.asyncio
async def test_share_absolute_path_outside_both_rejects(ark_home, tmp_path):
    """An absolute path that lands under neither the workspace nor the
    session's project root is rejected — the trust boundary stays scoped
    to places the agent legitimately operates in."""
    from ark import projects as _projects

    conn = db.init_db()
    workspace = tmp_path / "ws"; workspace.mkdir()
    project_root = tmp_path / "proj"; project_root.mkdir()
    p = _projects.create(
        conn, name="demo", root=str(project_root),
        description="", project_context="",
    )
    sid = runtime.create_session(conn, "scribe", "conversational", project_id=p.id)
    ctx = make_ctx(conn, workspace, session_id=sid)

    stray = tmp_path / "stray.txt"
    stray.write_text("not in either root")

    output, err = await tools.execute(
        "share_with_client", {"path": str(stray.resolve())}, ctx=ctx
    )
    assert err is True
    assert "outside" in output.lower()


@pytest.mark.asyncio
async def test_share_absolute_path_no_project_rejects_clearly(ark_home, tmp_path):
    """When the session isn't project-bound, an absolute path outside
    workspace fails with a message naming the specific limitation."""
    conn = db.init_db()
    workspace = tmp_path / "ws"; workspace.mkdir()
    sid = runtime.create_session(conn, "scribe", "conversational")
    ctx = make_ctx(conn, workspace, session_id=sid)

    stray = tmp_path / "stray.txt"
    stray.write_text("x")

    output, err = await tools.execute(
        "share_with_client", {"path": str(stray.resolve())}, ctx=ctx
    )
    assert err is True
    assert "no project binding" in output.lower()


def test_shared_file_round_trip_defaults_backwards_compat():
    """Legacy rows (no root/project_id) load as root='workspace' / None."""
    from ark.types import message_from_row

    restored = message_from_row(
        "shared_file",
        {"path": "chart.png", "description": "", "size": 42},
    )
    assert isinstance(restored, SharedFile)
    assert restored.root == "workspace"
    assert restored.project_id is None


def test_shared_file_round_trip_includes_project_fields_when_set():
    """New rows carry root + project_id in the serialized form."""
    from ark.types import message_from_row, message_to_row

    sf = SharedFile(
        path="Q3.pdf", description="", size=42,
        root="project", project_id="proj-uuid",
    )
    role, content = message_to_row(sf)
    assert content["root"] == "project"
    assert content["project_id"] == "proj-uuid"
    restored = message_from_row(role, content)
    assert restored.root == "project"
    assert restored.project_id == "proj-uuid"


def test_shared_file_wire_omits_defaults():
    """Workspace rows don't carry the extra fields — keeps content_json
    tiny for the common case (and leaves legacy on-disk rows untouched)."""
    from ark.types import message_to_row

    sf = SharedFile(path="chart.png", description="", size=42)
    _, content = message_to_row(sf)
    assert "root" not in content
    assert "project_id" not in content


# ---------------------------------------------------------------------------
# CLI: _download_shared routes to the right endpoint based on `root`
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cli_download_shared_workspace_hits_agent_files(tmp_path):
    """Default (root='workspace'): fetch via /agents/{agent}/files/<path>."""
    import httpx
    from ark.cli import _download_shared

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(200, content=b"hello")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        out = await _download_shared(http, "scribe", "chart.png", tmp_path)
    assert seen == ["/agents/scribe/files/chart.png"]
    assert out.read_bytes() == b"hello"


@pytest.mark.asyncio
async def test_cli_download_shared_project_hits_project_files(tmp_path):
    """root='project' + project_id: fetch via /projects/{pid}/files/<path>,
    not the agent-workspace endpoint (which would 404)."""
    import httpx
    from ark.cli import _download_shared

    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        return httpx.Response(200, content=b"pdfbytes")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        out = await _download_shared(
            http, "scribe", "Q3-report.pdf", tmp_path,
            root="project", project_id="abc-123",
        )
    assert seen == ["/projects/abc-123/files/Q3-report.pdf"]
    assert out.read_bytes() == b"pdfbytes"


@pytest.mark.asyncio
async def test_cli_download_shared_project_without_id_errors(tmp_path):
    """A root='project' frame with no project_id is a malformed event; the
    client refuses rather than silently falling back to the workspace URL."""
    import httpx
    from ark.cli import _download_shared

    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b""))
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        with pytest.raises(ValueError, match="project_id"):
            await _download_shared(
                http, "scribe", "x.md", tmp_path, root="project", project_id=None,
            )


# ---------------------------------------------------------------------------
# Provider folding for the new message types
# ---------------------------------------------------------------------------


def test_anthropic_folds_upload_as_user_text():
    from ark.types import UserText

    out = to_anthropic_messages(
        [
            UploadMessage(path="uploads/r.pdf", original_name="report.pdf", size=42),
            UserText(text="summarize please"),
        ]
    )
    assert out[0]["role"] == "user"
    assert "report.pdf" in out[0]["content"]
    assert "uploads/r.pdf" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "summarize please"}


def test_anthropic_folds_shared_file_into_assistant():
    from ark.types import AssistantText

    out = to_anthropic_messages(
        [
            AssistantText(text="here you go"),
            SharedFile(path="chart.png", description="Q4 summary", size=100),
        ]
    )
    assert out[0]["role"] == "assistant"
    text_blocks = [b for b in out[0]["content"] if b["type"] == "text"]
    assert any("here you go" in b["text"] for b in text_blocks)
    assert any("chart.png" in b["text"] and "Q4 summary" in b["text"] for b in text_blocks)


def test_openai_folds_upload_and_shared_file():
    from ark.types import AssistantText, UserText

    out = to_openai_messages(
        "system",
        [
            UploadMessage(path="uploads/r.pdf", original_name="report.pdf", size=42),
            UserText(text="summarize"),
            AssistantText(text="done"),
            SharedFile(path="summary.md", size=200),
        ],
    )
    # system + user(upload) + user(text) + assistant(text + shared)
    assert out[0] == {"role": "system", "content": "system"}
    assert "report.pdf" in out[1]["content"]
    assert out[2] == {"role": "user", "content": "summarize"}
    assert out[3]["role"] == "assistant"
    assert "done" in out[3]["content"]
    assert "summary.md" in out[3]["content"]
