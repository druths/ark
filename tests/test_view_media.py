"""Phase 1 multimodal: inbound images via the `view_media` tool.

Covers:
- Tool contract: validates file exists, detects mime, rejects unsupported
  types, registers attachments on ctx.pending_attachments.
- ToolResult carries attachments through persistence + round-trip.
- All three provider adapters translate ToolResult-with-image into the
  correct native wire shape (Anthropic image block in tool_result content;
  OpenAI image_url in follow-up user message; Gemini inline_data Part in
  follow-up user Content).
- End-to-end runtime: view_media call → ToolResult row has attachments →
  message list sent to provider includes the image block.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ark import db, runtime, tools
from ark.config import AgentConfig, Config, ProviderConfig, ServerConfig
from ark.provider import (
    _anthropic_image_block,
    _google_image_part,
    _openai_image_block,
    to_anthropic_messages,
    to_google_contents,
    to_openai_messages,
)
from ark.tools import ToolContext
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    TextDelta,
    ToolCall,
    ToolResult,
    TurnUsageEvent,
    UserText,
    message_from_row,
    message_to_row,
)


# Tiny valid 1x1 PNG (base64-decoded here for use in tests).
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII="
)
# Tiny valid 1x1 JPEG.
_TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD/AP/Z"
)


def _make_ctx(tmp_path, agent_name="scribe"):
    cwd = tmp_path / "ws"
    cwd.mkdir(exist_ok=True)
    agent = AgentConfig(
        name=agent_name, provider="a", model="claude-sonnet-4-6", workspace=cwd
    )
    return ToolContext(
        conn=MagicMock(),
        config=MagicMock(),
        agent=agent,
        session_id="s",
        cwd=cwd,
        loaded_skills=set(),
    )


def _run(name, args, ctx):
    return asyncio.run(tools.execute(name, args, ctx=ctx))


# ---------------------------------------------------------------------------
# view_media tool contract
# ---------------------------------------------------------------------------


def test_view_media_registers_png_attachment(tmp_path):
    ctx = _make_ctx(tmp_path)
    img = ctx.cwd / "photo.png"
    img.write_bytes(_TINY_PNG)

    output, err = _run("view_media", {"path": str(img)}, ctx)
    assert err is False
    assert "attached image" in output
    # execute() drains pending_attachments only when called from
    # run_user_turn; the ctx still holds it after the tool returns here.
    assert ctx.pending_attachments is not None
    assert len(ctx.pending_attachments) == 1
    att = ctx.pending_attachments[0]
    assert att["type"] == "image"
    assert att["mime"] == "image/png"
    assert att["path"] == str(img.resolve())


def test_view_media_detects_jpeg_by_magic_bytes(tmp_path):
    ctx = _make_ctx(tmp_path)
    # Save a JPEG with a misleading .png extension — magic bytes should win.
    img = ctx.cwd / "misnamed.png"
    img.write_bytes(_TINY_JPEG)
    output, err = _run("view_media", {"path": str(img)}, ctx)
    assert err is False
    att = ctx.pending_attachments[0]
    assert att["mime"] == "image/jpeg"


def test_view_media_rejects_missing_file(tmp_path):
    ctx = _make_ctx(tmp_path)
    output, err = _run("view_media", {"path": str(tmp_path / "nope.png")}, ctx)
    assert err is True
    assert "not a file" in output
    assert not ctx.pending_attachments  # nothing registered


def test_view_media_rejects_unsupported_mime_for_v1(tmp_path):
    """PDFs, audio, video are explicitly out-of-scope for v1 — fail with a
    clear error pointing at the capability horizon."""
    ctx = _make_ctx(tmp_path)
    pdf = ctx.cwd / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n...")  # realistic enough for magic/ext detection
    output, err = _run("view_media", {"path": str(pdf)}, ctx)
    assert err is True
    assert "unsupported" in output.lower()
    assert not ctx.pending_attachments


# ---------------------------------------------------------------------------
# ToolResult carries attachments through persistence
# ---------------------------------------------------------------------------


def test_tool_result_attachments_round_trip():
    tr = ToolResult(
        call_id="t1", output="attached", is_error=False, name="view_media",
        attachments=[{"type": "image", "path": "/tmp/x.png", "mime": "image/png"}],
    )
    role, content = message_to_row(tr)
    assert content["attachments"][0]["type"] == "image"
    restored = message_from_row(role, content)
    assert isinstance(restored, ToolResult)
    assert restored.attachments[0]["mime"] == "image/png"


def test_tool_result_empty_attachments_omitted_from_wire():
    """Backwards-compat: an attachment-free ToolResult shouldn't carry the
    key in content_json (keeps rows tiny for the common case)."""
    tr = ToolResult(call_id="t1", output="done", is_error=False, name="x")
    _, content = message_to_row(tr)
    assert "attachments" not in content


def test_tool_result_round_trip_defaults_to_empty_attachments():
    """Loading a legacy row (no attachments key) yields an empty list."""
    restored = message_from_row(
        "tool_result", {"call_id": "t1", "output": "done"}
    )
    assert isinstance(restored, ToolResult)
    assert restored.attachments == []


# ---------------------------------------------------------------------------
# Shared image loader + per-provider block builders
# ---------------------------------------------------------------------------


def test_anthropic_image_block_round_trip(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    block = _anthropic_image_block(
        {"type": "image", "path": str(img), "mime": "image/png"}
    )
    assert block["type"] == "image"
    assert block["source"]["type"] == "base64"
    assert block["source"]["media_type"] == "image/png"
    assert base64.b64decode(block["source"]["data"]) == _TINY_PNG


def test_openai_image_block_round_trip(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    block = _openai_image_block(
        {"type": "image", "path": str(img), "mime": "image/png"}
    )
    assert block["type"] == "image_url"
    url = block["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == _TINY_PNG


def test_google_image_part_round_trip(tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    part = _google_image_part(
        {"type": "image", "path": str(img), "mime": "image/png"}
    )
    # Part shape is pydantic — we check inline_data fields are populated.
    assert part.inline_data is not None
    assert part.inline_data.mime_type == "image/png"
    assert part.inline_data.data == _TINY_PNG


def test_image_block_builders_return_none_for_missing_file(tmp_path):
    att = {"type": "image", "path": str(tmp_path / "nope.png"), "mime": "image/png"}
    assert _anthropic_image_block(att) is None
    assert _openai_image_block(att) is None
    assert _google_image_part(att) is None


def test_image_block_builders_return_none_for_non_image_type():
    """Phase 1 only knows images; PDF/audio/video attachments return None
    (adapters silently drop them from the message list)."""
    att = {"type": "pdf", "path": "/tmp/x.pdf", "mime": "application/pdf"}
    assert _anthropic_image_block(att) is None
    assert _openai_image_block(att) is None
    assert _google_image_part(att) is None


# ---------------------------------------------------------------------------
# Adapter message-list shapes
# ---------------------------------------------------------------------------


def test_anthropic_tool_result_with_image_uses_content_array(tmp_path):
    """Anthropic's tool_result supports a content ARRAY that mixes text +
    image blocks (newer Claude feature). We use that shape when attachments
    are present; stay with bare-string content for pure-text results."""
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    messages = [
        ToolCall(id="t1", name="view_media", input={"path": str(img)}),
        ToolResult(
            call_id="t1", output="attached", is_error=False, name="view_media",
            attachments=[{"type": "image", "path": str(img), "mime": "image/png"}],
        ),
    ]
    out = to_anthropic_messages(messages)
    tool_result_msg = next(
        m for m in out if m["role"] == "user" and isinstance(m["content"], list)
    )
    block = tool_result_msg["content"][0]
    assert block["type"] == "tool_result"
    # Content is a list of parts, first is text, second is image.
    content = block["content"]
    assert isinstance(content, list)
    kinds = [p["type"] for p in content]
    assert "text" in kinds and "image" in kinds


def test_anthropic_tool_result_without_image_stays_bare_string(tmp_path):
    """Backwards-compat: text-only ToolResults keep the old wire shape
    (bare `content: str`) — don't regress what already works."""
    messages = [
        ToolCall(id="t1", name="read_file", input={"path": "x"}),
        ToolResult(call_id="t1", output="file contents", is_error=False, name="read_file"),
    ]
    out = to_anthropic_messages(messages)
    tool_result_msg = next(
        m for m in out if m["role"] == "user" and isinstance(m["content"], list)
    )
    block = tool_result_msg["content"][0]
    assert block["content"] == "file contents"  # bare string, not a list


def test_openai_tool_result_with_image_emits_follow_up_user_message(tmp_path):
    """OpenAI's `tool` role content is text-only, so an image attached to
    a tool result ships as a follow-up user message with image_url blocks.
    OpenRouter translates this to the routed model's native shape."""
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    messages = [
        ToolCall(id="t1", name="view_media", input={"path": str(img)}),
        ToolResult(
            call_id="t1", output="attached", is_error=False, name="view_media",
            attachments=[{"type": "image", "path": str(img), "mime": "image/png"}],
        ),
    ]
    out = to_openai_messages("sys", messages)
    tool_idx = next(i for i, m in enumerate(out) if m.get("role") == "tool")
    follow_up = out[tool_idx + 1]
    assert follow_up["role"] == "user"
    kinds = [p["type"] for p in follow_up["content"]]
    assert "text" in kinds and "image_url" in kinds
    # The text pointer mentions the tool_call_id for correlation.
    text_part = next(p for p in follow_up["content"] if p["type"] == "text")
    assert "t1" in text_part["text"]


def test_openai_tool_result_without_image_emits_no_follow_up(tmp_path):
    messages = [
        ToolCall(id="t1", name="read_file", input={"path": "x"}),
        ToolResult(call_id="t1", output="content", is_error=False, name="read_file"),
    ]
    out = to_openai_messages("sys", messages)
    tool_idx = next(i for i, m in enumerate(out) if m.get("role") == "tool")
    # No follow-up user message (just the tool message, possibly end of list).
    if tool_idx + 1 < len(out):
        assert out[tool_idx + 1].get("role") != "user"


def test_google_tool_result_with_image_emits_follow_up_user_content(tmp_path):
    """Gemini's FunctionResponse is text-only, so images attached to a
    tool result ship as a follow-up user Content with inline_data parts."""
    img = tmp_path / "x.png"
    img.write_bytes(_TINY_PNG)
    messages = [
        ToolCall(id="t1", name="view_media", input={"path": str(img)}),
        ToolResult(
            call_id="t1", output="attached", is_error=False, name="view_media",
            attachments=[{"type": "image", "path": str(img), "mime": "image/png"}],
        ),
    ]
    out = to_google_contents(messages)
    # Last Content should be the follow-up user with text + image parts.
    last = out[-1]
    assert last.role == "user"
    kinds = [
        "text" if p.text is not None
        else ("inline_data" if p.inline_data is not None else "other")
        for p in last.parts
    ]
    assert "text" in kinds and "inline_data" in kinds


# ---------------------------------------------------------------------------
# End-to-end: runtime drains pending_attachments into ToolResult
# ---------------------------------------------------------------------------


def _cfg(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return Config(
        server=ServerConfig(host="127.0.0.1", port=7777, auth_secret="x"),
        providers={"a": ProviderConfig(provider_type="anthropic", api_key="k")},
        tools={},
        agents={
            "scribe": AgentConfig(
                name="scribe", provider="a", model="claude-sonnet-4-6",
                workspace=ws,
            )
        },
    )


@pytest.mark.asyncio
async def test_runtime_drains_attachments_into_tool_result(ark_home, tmp_path):
    """End-to-end: a model that calls view_media → the ToolResult row
    persisted after that call carries the attachment, and the next
    iteration's provider call sees the image in its message list."""
    cfg = _cfg(tmp_path)
    conn = db.init_db()
    sid = runtime.create_session(conn, "scribe", "conversational")

    img = cfg.agents["scribe"].workspace / "mockup.png"
    img.write_bytes(_TINY_PNG)

    captured_messages: list = []

    class _StubProvider:
        def __init__(self):
            self.calls = 0

        async def stream_turn(self, *, model, system, messages, tools, max_tokens=4096):
            self.calls += 1
            captured_messages.append(list(messages))
            if self.calls == 1:
                # First iteration: model asks to view the image.
                from ark.types import ToolCallEvent
                yield TurnUsageEvent(input_tokens=50, output_tokens=10, model=model)
                yield ToolCallEvent(
                    id="t1", name="view_media", input={"path": str(img)}
                )
                yield AssistantTurnEnd(text="", stop_reason="tool_use")
            else:
                # Second iteration: model describes what it saw.
                yield TurnUsageEvent(input_tokens=80, output_tokens=15, model=model)
                yield TextDelta(text="it's a tiny 1x1 PNG")
                yield AssistantTurnEnd(
                    text="it's a tiny 1x1 PNG", stop_reason="end_turn"
                )

    async for _ in runtime.run_user_turn(
        conn=conn, config=cfg, agent=cfg.agents["scribe"],
        session_id=sid, user_text="look at mockup.png",
        provider_factory=lambda *_a, **_k: _StubProvider(),
    ):
        pass

    # The ToolResult row has the attachment persisted.
    history = runtime.load_history(conn, sid)
    tool_results = [m for m in history if isinstance(m, ToolResult)]
    assert len(tool_results) == 1
    assert len(tool_results[0].attachments) == 1
    assert tool_results[0].attachments[0]["mime"] == "image/png"

    # The second iteration's message list contains the ToolResult with
    # attachments — the adapter will translate it to the provider-native
    # image block when the real stream_turn runs. (Our stub just captures
    # the Message list before translation.)
    second_turn = captured_messages[1]
    tr_in_list = next(m for m in second_turn if isinstance(m, ToolResult))
    assert tr_in_list.attachments == tool_results[0].attachments
