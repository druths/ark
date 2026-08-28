"""OpenAI provider: message folding + stream translation."""

from types import SimpleNamespace

import pytest

from ark.provider import _translate_openai_stream, to_openai_messages, to_openai_tool
from ark.types import (
    AssistantText,
    AssistantTurnEnd,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolSchema,
    UserText,
)


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def test_fold_user_message():
    assert to_openai_messages("", [UserText(text="hi")]) == [
        {"role": "user", "content": "hi"}
    ]


def test_fold_system_prepended():
    out = to_openai_messages("be helpful", [UserText(text="hi")])
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_fold_assistant_text_and_tool_call():
    out = to_openai_messages(
        "",
        [
            UserText(text="?"),
            AssistantText(text="sure"),
            ToolCall(id="t1", name="read_file", input={"path": "/x"}),
        ],
    )
    assert out[1] == {
        "role": "assistant",
        "content": "sure",
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "/x"}'},
            }
        ],
    }


def test_fold_tool_result_role():
    out = to_openai_messages("", [ToolResult(call_id="t1", output="contents")])
    assert out == [{"role": "tool", "tool_call_id": "t1", "content": "contents"}]


def test_fold_tool_result_error_prefixed():
    out = to_openai_messages("", [ToolResult(call_id="t1", output="boom", is_error=True)])
    assert out[0]["content"].startswith("ERROR:")


def test_tool_schema_conversion():
    s = ToolSchema(
        name="read_file",
        description="read",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    )
    assert to_openai_tool(s) == {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "read",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }


# ---------------------------------------------------------------------------
# Stream translation
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()


@pytest.mark.asyncio
async def test_translate_text_and_tool_call_stream():
    from ark.types import TurnUsageEvent

    chunks = [
        ns(choices=[ns(delta=ns(content="Hi", tool_calls=None), finish_reason=None)], usage=None),
        ns(choices=[ns(delta=ns(content=" there", tool_calls=None), finish_reason=None)], usage=None),
        ns(
            choices=[
                ns(
                    delta=ns(
                        content=None,
                        tool_calls=[
                            ns(
                                index=0,
                                id="call_1",
                                function=ns(name="read_file", arguments='{"path":'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        ns(
            choices=[
                ns(
                    delta=ns(
                        content=None,
                        tool_calls=[
                            ns(
                                index=0,
                                id=None,
                                function=ns(name=None, arguments=' "a.txt"}'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        ns(
            choices=[
                ns(
                    delta=ns(content=None, tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
        # Final usage chunk has no choices, only `usage` (when stream_options.include_usage is set)
        ns(choices=[], usage=ns(prompt_tokens=42, completion_tokens=7)),
    ]
    out = []
    async for evt in _translate_openai_stream(_FakeStream(chunks), model="gpt-5"):
        out.append(evt)
    # Strip usage for the existing assertion shape
    out_no_usage = [e for e in out if not isinstance(e, TurnUsageEvent)]
    assert out_no_usage == [
        TextDelta(text="Hi"),
        TextDelta(text=" there"),
        ToolCallEvent(id="call_1", name="read_file", input={"path": "a.txt"}),
        AssistantTurnEnd(text="Hi there", stop_reason="tool_calls"),
    ]
    # And the usage event must be present, populated from the final chunk
    usage = [e for e in out if isinstance(e, TurnUsageEvent)]
    assert len(usage) == 1
    assert usage[0].input_tokens == 42
    assert usage[0].output_tokens == 7
    assert usage[0].model == "gpt-5"


def test_usage_chunk_cached_details_parsed():
    """mine-capstone#697: prompt_tokens is already the total; the details
    name the cached share (and OpenRouter forwards cache_creation)."""
    import asyncio

    from ark.provider import _translate_openai_stream
    from ark.types import TurnUsageEvent

    ns = SimpleNamespace

    async def stream():
        yield ns(choices=[ns(delta=ns(content="ok", tool_calls=None), finish_reason="stop")], usage=None)
        yield ns(
            choices=[],
            usage=ns(
                prompt_tokens=5_000, completion_tokens=9,
                prompt_tokens_details=ns(cached_tokens=4_200),
                cache_creation_input_tokens=300,
            ),
        )

    async def run():
        return [e async for e in _translate_openai_stream(stream(), model="anthropic/claude-sonnet-4-6")]

    out = asyncio.run(run())
    (usage,) = [e for e in out if isinstance(e, TurnUsageEvent)]
    assert usage.input_tokens == 5_000
    assert usage.cached_input_tokens == 4_200
    assert usage.cache_write_tokens == 300
