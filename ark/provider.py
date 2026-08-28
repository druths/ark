"""LLM provider abstraction + Anthropic adapter.

Providers translate Ark's normalized `Message` / `ToolSchema` types into the
underlying API's wire format, and the streamed responses back into the
normalized `ProviderEvent` union.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable, Protocol

import anthropic
from google import genai
from google.genai import types as gtypes
from openai import AsyncOpenAI

from .types import (
    AssistantText,
    AssistantTurnEnd,
    Message,
    ProviderEvent,
    SharedFile,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolSchema,
    TurnUsageEvent,
    UploadMessage,
    UserText,
)


def _upload_marker(msg: UploadMessage) -> str:
    return (
        f"[user attached a file: original name '{msg.original_name}', "
        f"workspace path '{msg.path}', {msg.size} bytes]"
    )


def _shared_marker(msg: SharedFile) -> str:
    desc = f' — "{msg.description}"' if msg.description else ""
    return f"[shared with user: '{msg.path}' ({msg.size} bytes){desc}]"


class Provider(Protocol):
    async def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
        prompt_caching: bool = False,
    ) -> AsyncIterator[ProviderEvent]: ...


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class AnthropicProvider:
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
        prompt_caching: bool = False,
    ) -> AsyncIterator[ProviderEvent]:
        api_messages = to_anthropic_messages(messages)
        api_tools = [to_anthropic_tool(t) for t in tools]
        api_system: Any = system
        if prompt_caching:
            # cache_control on the stable prefix (system) + the tail markers
            # (last user-text block, last block) — the Anthropic recipe for
            # a conversation whose prefix repeats every call.
            api_system = [
                {"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}
            ]
            _mark_anthropic_cache(api_messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "system": api_system,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if api_tools:
            kwargs["tools"] = api_tools
        state = _State()
        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                for translated in _translate(event, state, model):
                    yield translated


class _State:
    __slots__ = (
        "block_type",
        "tool_id",
        "tool_name",
        "tool_json",
        "text_buf",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
    )

    def __init__(self) -> None:
        self.block_type: str | None = None
        self.tool_id: str | None = None
        self.tool_name: str | None = None
        self.tool_json: str = ""
        self.text_buf: str = ""
        self.stop_reason: str | None = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cached_input_tokens: int = 0
        self.cache_write_tokens: int = 0


def _blockify(content: Any) -> list[dict[str, Any]]:
    """A message's content as a block list (cache_control needs blocks)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content)


def _mark_anthropic_cache(api_messages: list[dict[str, Any]]) -> None:
    """cache_control markers on the tail: the last user-TEXT block and the
    last block overall — with the system marker, the whole repeated prefix
    is cacheable and only the newest turn misses."""
    if not api_messages:
        return
    last_user_text = None
    for msg in reversed(api_messages):
        if msg.get("role") != "user":
            continue
        blocks = _blockify(msg["content"])
        for block in reversed(blocks):
            if isinstance(block, dict) and block.get("type") == "text":
                last_user_text = (msg, blocks, block)
                break
        if last_user_text:
            break
    if last_user_text:
        msg, blocks, block = last_user_text
        block["cache_control"] = {"type": "ephemeral"}
        msg["content"] = blocks
    last = api_messages[-1]
    blocks = _blockify(last["content"])
    tail = blocks[-1]
    if isinstance(tail, dict) and "cache_control" not in tail:
        tail["cache_control"] = {"type": "ephemeral"}
    last["content"] = blocks


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Fold our flat message list into Anthropic's nested role/content shape.

    Consecutive `AssistantText` + `ToolCall` entries collapse into a single
    assistant message with multiple content blocks. Consecutive `ToolResult`
    entries collapse into a single user message with `tool_result` blocks.
    """

    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if isinstance(m, UserText):
            out.append({"role": "user", "content": m.text})
            i += 1
        elif isinstance(m, UploadMessage):
            out.append({"role": "user", "content": _upload_marker(m)})
            i += 1
        elif isinstance(m, (AssistantText, ToolCall, SharedFile)):
            blocks: list[dict[str, Any]] = []
            while i < n and isinstance(messages[i], (AssistantText, ToolCall, SharedFile)):
                msg = messages[i]
                if isinstance(msg, AssistantText):
                    if msg.text:
                        blocks.append({"type": "text", "text": msg.text})
                elif isinstance(msg, SharedFile):
                    blocks.append({"type": "text", "text": _shared_marker(msg)})
                else:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": msg.id,
                            "name": msg.name,
                            "input": msg.input,
                        }
                    )
                i += 1
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolResult):
            blocks = []
            while i < n and isinstance(messages[i], ToolResult):
                tr = messages[i]
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tr.call_id,
                    "content": tr.output,
                }
                if tr.is_error:
                    block["is_error"] = True
                blocks.append(block)
                i += 1
            out.append({"role": "user", "content": blocks})
        else:  # pragma: no cover
            raise TypeError(f"unknown message: {type(m).__name__}")
    return out


def to_anthropic_tool(t: ToolSchema) -> dict[str, Any]:
    return {
        "name": t.name,
        "description": t.description,
        "input_schema": t.input_schema,
    }


def translate_stream(events: Iterable[Any], model: str = "") -> Iterable[ProviderEvent]:
    """Convert raw Anthropic stream events into normalized ProviderEvents.

    Pulled out as a sync helper over a regular iterable so it can be unit-tested
    without the async SDK.
    """

    state = _State()
    for event in events:
        yield from _translate(event, state, model)


def _translate(event: Any, state: _State, model: str = "") -> Iterable[ProviderEvent]:
    et = getattr(event, "type", None)
    if et == "message_start":
        # Initial usage (input tokens) lives on the message_start envelope.
        msg = getattr(event, "message", None)
        usage = getattr(msg, "usage", None) if msg is not None else None
        if usage is not None:
            uncached = int(getattr(usage, "input_tokens", 0) or 0)
            cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
            written = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
            # Normalize input_tokens to the TOTAL prompt — Anthropic's raw
            # field excludes cached/creation tokens when caching is on.
            state.input_tokens = uncached + cached + written
            state.cached_input_tokens = cached
            state.cache_write_tokens = written
            state.output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    elif et == "content_block_start":
        cb = getattr(event, "content_block", None)
        if cb is None:
            return
        state.block_type = getattr(cb, "type", None)
        if state.block_type == "tool_use":
            state.tool_id = getattr(cb, "id", None)
            state.tool_name = getattr(cb, "name", None)
            state.tool_json = ""
    elif et == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta is None:
            return
        dt = getattr(delta, "type", None)
        if dt == "text_delta":
            text = getattr(delta, "text", "")
            state.text_buf += text
            if text:
                yield TextDelta(text=text)
        elif dt == "thinking_delta":
            text = getattr(delta, "thinking", "")
            if text:
                yield ThinkingDelta(text=text)
        elif dt == "input_json_delta":
            state.tool_json += getattr(delta, "partial_json", "")
        # signature_delta and unknowns: ignore
    elif et == "content_block_stop":
        if state.block_type == "tool_use" and state.tool_id is not None:
            parsed: dict[str, Any] = {}
            if state.tool_json.strip():
                try:
                    parsed = json.loads(state.tool_json)
                except json.JSONDecodeError:
                    parsed = {}
            yield ToolCallEvent(
                id=state.tool_id,
                name=state.tool_name or "",
                input=parsed,
            )
            state.tool_id = None
            state.tool_name = None
            state.tool_json = ""
        state.block_type = None
    elif et == "message_delta":
        delta = getattr(event, "delta", None)
        if delta is not None:
            sr = getattr(delta, "stop_reason", None)
            if sr:
                state.stop_reason = sr
        # Output-tokens cumulative count lives on usage of the message_delta
        # event itself (not on `delta`).
        usage = getattr(event, "usage", None)
        if usage is not None:
            out = getattr(usage, "output_tokens", None)
            if out is not None:
                state.output_tokens = int(out)
    elif et == "message_stop":
        yield TurnUsageEvent(
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            model=model,
            cached_input_tokens=state.cached_input_tokens,
            cache_write_tokens=state.cache_write_tokens,
        )
        yield AssistantTurnEnd(text=state.text_buf, stop_reason=state.stop_reason)
        state.text_buf = ""
        state.stop_reason = None
        state.input_tokens = 0
        state.output_tokens = 0


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIProvider:
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncOpenAI(**kwargs)

    async def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
        prompt_caching: bool = False,
    ) -> AsyncIterator[ProviderEvent]:
        api_messages = self._prepare_messages(
            system, messages, model=model, prompt_caching=prompt_caching
        )
        api_tools = [to_openai_tool(t) for t in tools] or None
        # `max_completion_tokens` is the modern field; reasoning models (gpt-5, o1,
        # o3) reject `max_tokens` outright. Older models accept both.
        # `stream_options.include_usage` gets us token counts in a final chunk.
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if api_tools:
            kwargs["tools"] = api_tools
        stream = await self._client.chat.completions.create(**kwargs)
        async for evt in _translate_openai_stream(stream, model=model):
            yield evt


    def _prepare_messages(
        self, system: str, messages: list[Message], *, model: str,
        prompt_caching: bool,
    ) -> list[dict[str, Any]]:
        # Base OpenAI: providers cache implicitly (prompt_tokens_details
        # reports the hits) — no request-side markers exist.
        return to_openai_messages(system, messages)


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter via the OpenAI-compatible chat completions API.

    Model ids are namespaced by source provider — e.g. `anthropic/claude-sonnet-4-6`,
    `openai/gpt-4o`, `meta-llama/llama-3.1-70b-instruct`. See
    https://openrouter.ai/models for the full catalog.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or self.DEFAULT_BASE_URL)

    def _prepare_messages(
        self, system: str, messages: list[Message], *, model: str,
        prompt_caching: bool,
    ) -> list[dict[str, Any]]:
        api_messages = to_openai_messages(system, messages)
        # Anthropic models behind OpenRouter take the SAME cache_control
        # markers, carried as multipart content. Never for other sources
        # (moonshotai/* et al. reject or ignore unknown part keys — and
        # Moonshot's caching is implicit anyway).
        if prompt_caching and model.startswith("anthropic/"):
            _mark_openrouter_cache(api_messages)
        return api_messages


def _mark_openrouter_cache(api_messages: list[dict[str, Any]]) -> None:
    """cache_control on the system part + the last user text part —
    OpenRouter forwards the markers to Anthropic when content is multipart."""

    def to_parts(msg: dict[str, Any]) -> list[dict[str, Any]]:
        content = msg["content"]
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return list(content)

    for msg in api_messages:
        if msg.get("role") == "system":
            parts = to_parts(msg)
            if parts and isinstance(parts[-1], dict):
                parts[-1]["cache_control"] = {"type": "ephemeral"}
            msg["content"] = parts
            break
    for msg in reversed(api_messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), (str, list)):
            parts = to_parts(msg)
            for part in reversed(parts):
                if isinstance(part, dict) and part.get("type") == "text":
                    part["cache_control"] = {"type": "ephemeral"}
                    break
            msg["content"] = parts
            break


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Fold flat messages into OpenAI chat-completions shape."""

    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if isinstance(m, UserText):
            out.append({"role": "user", "content": m.text})
            i += 1
        elif isinstance(m, UploadMessage):
            out.append({"role": "user", "content": _upload_marker(m)})
            i += 1
        elif isinstance(m, (AssistantText, ToolCall, SharedFile)):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            while i < n and isinstance(messages[i], (AssistantText, ToolCall, SharedFile)):
                msg = messages[i]
                if isinstance(msg, AssistantText):
                    if msg.text:
                        text_parts.append(msg.text)
                elif isinstance(msg, SharedFile):
                    text_parts.append(_shared_marker(msg))
                else:
                    tool_calls.append(
                        {
                            "id": msg.id,
                            "type": "function",
                            "function": {
                                "name": msg.name,
                                "arguments": json.dumps(msg.input),
                            },
                        }
                    )
                i += 1
            entry: dict[str, Any] = {"role": "assistant"}
            entry["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        elif isinstance(m, ToolResult):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.call_id,
                    "content": m.output if not m.is_error else f"ERROR: {m.output}",
                }
            )
            i += 1
        else:  # pragma: no cover
            raise TypeError(f"unknown message: {type(m).__name__}")
    return out


def to_openai_tool(t: ToolSchema) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_schema,
        },
    }


async def _translate_openai_stream(stream, model: str = "") -> AsyncIterator[ProviderEvent]:
    """Convert OpenAI streaming chunks to normalized ProviderEvents.

    OpenAI tool-call deltas come keyed by `index`; we assemble them up before
    emitting the ToolCallEvent at end-of-stream. Usage arrives on a final
    chunk when `stream_options.include_usage` was requested.
    """

    text_buf = ""
    stop_reason: str | None = None
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    cache_write_tokens = 0
    # index -> {id, name, args}
    tool_calls: dict[int, dict[str, Any]] = {}

    async for chunk in stream:
        # The usage chunk has no choices and is emitted after the content.
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            # OpenAI-compatible cache telemetry: prompt_tokens is already
            # the TOTAL; details name the cached share. OpenRouter forwards
            # Anthropic's cache_creation as an extra field when present.
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)
            cache_write_tokens = int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            )
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            text_buf += delta.content
            yield TextDelta(text=delta.content)
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                slot = tool_calls.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments
        if choice.finish_reason:
            stop_reason = choice.finish_reason

    for idx in sorted(tool_calls):
        slot = tool_calls[idx]
        parsed: dict[str, Any] = {}
        if slot["args"].strip():
            try:
                parsed = json.loads(slot["args"])
            except json.JSONDecodeError:
                parsed = {}
        yield ToolCallEvent(
            id=slot["id"] or f"call_{idx}",
            name=slot["name"] or "",
            input=parsed,
        )
    yield TurnUsageEvent(
        input_tokens=input_tokens, output_tokens=output_tokens, model=model,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    yield AssistantTurnEnd(text=text_buf, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Google (Gemini) adapter
# ---------------------------------------------------------------------------


class GoogleProvider:
    """Native Google Gemini adapter using the `google-genai` SDK.

    Differences worth knowing vs the other adapters:
    - Roles are `user` and `model` (not `assistant`).
    - The system prompt is a separate config field, not a message.
    - Tool calls come back as `function_call` parts inside a single Content;
      they aren't split across chunks the way OpenAI's tool-call arguments are.
    - We disable `automatic_function_calling` so the SDK yields the tool
      calls back to us instead of trying to dispatch them itself.
    """

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["http_options"] = gtypes.HttpOptions(base_url=base_url)
        self._client = genai.Client(**client_kwargs)

    async def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        tools: list[ToolSchema],
        max_tokens: int = 4096,
        prompt_caching: bool = False,  # Google caches implicitly; reported only
    ) -> AsyncIterator[ProviderEvent]:
        contents = to_google_contents(messages)
        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": gtypes.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if system:
            config_kwargs["system_instruction"] = system
        if tools:
            config_kwargs["tools"] = [
                gtypes.Tool(
                    function_declarations=[to_google_function_declaration(t) for t in tools]
                )
            ]
        config = gtypes.GenerateContentConfig(**config_kwargs)
        stream = await self._client.aio.models.generate_content_stream(
            model=model, contents=contents, config=config
        )
        async for evt in _translate_google_stream(stream, model=model):
            yield evt


def to_google_contents(messages: list[Message]) -> list[Any]:
    """Fold our flat message list into Google's `Content` shape.

    user-side cluster (UserText, UploadMessage) → role="user", text parts.
    assistant-side cluster (AssistantText, ToolCall, SharedFile) → role="model".
    ToolResult cluster → role="user", function_response parts.
    """

    out: list[Any] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if isinstance(m, (UserText, UploadMessage)):
            parts: list[Any] = []
            while i < n and isinstance(messages[i], (UserText, UploadMessage)):
                msg = messages[i]
                text = msg.text if isinstance(msg, UserText) else _upload_marker(msg)
                if text:
                    parts.append(gtypes.Part(text=text))
                i += 1
            if parts:
                out.append(gtypes.Content(role="user", parts=parts))
        elif isinstance(m, (AssistantText, ToolCall, SharedFile)):
            parts = []
            while i < n and isinstance(messages[i], (AssistantText, ToolCall, SharedFile)):
                msg = messages[i]
                if isinstance(msg, AssistantText):
                    if msg.text:
                        parts.append(gtypes.Part(text=msg.text))
                elif isinstance(msg, SharedFile):
                    parts.append(gtypes.Part(text=_shared_marker(msg)))
                else:
                    part_kwargs: dict[str, Any] = {
                        "function_call": gtypes.FunctionCall(
                            id=msg.id, name=msg.name, args=msg.input or {}
                        )
                    }
                    if msg.thought_signature:
                        part_kwargs["thought_signature"] = msg.thought_signature
                    parts.append(gtypes.Part(**part_kwargs))
                i += 1
            if parts:
                out.append(gtypes.Content(role="model", parts=parts))
        elif isinstance(m, ToolResult):
            parts = []
            while i < n and isinstance(messages[i], ToolResult):
                tr = messages[i]
                # Google's API requires a non-empty `name` on every FunctionResponse.
                # ToolResult carries it directly on new rows; for old rows that
                # predate that field, look back for the matching ToolCall by id.
                name = tr.name or _lookup_tool_name(messages, i, tr.call_id)
                response: dict[str, Any] = (
                    {"error": tr.output} if tr.is_error else {"output": tr.output}
                )
                parts.append(
                    gtypes.Part(
                        function_response=gtypes.FunctionResponse(
                            id=tr.call_id,
                            name=name,
                            response=response,
                        )
                    )
                )
                i += 1
            out.append(gtypes.Content(role="user", parts=parts))
        else:  # pragma: no cover
            raise TypeError(f"unknown message: {type(m).__name__}")
    return out


def to_google_function_declaration(t: ToolSchema) -> Any:
    """Convert our ToolSchema to a Google FunctionDeclaration."""

    return gtypes.FunctionDeclaration(
        name=t.name,
        description=t.description,
        parameters_json_schema=t.input_schema,
    )


def _lookup_tool_name(messages: list[Message], idx: int, call_id: str) -> str:
    """Find the name of the ToolCall that produced the ToolResult at `idx`.

    Walks backward through the message list, matching by call_id. Used when
    folding for Google's API (which requires non-empty `name` on
    FunctionResponse) and the ToolResult itself lacks it — old rows from
    before ToolResult.name was added.
    """

    for j in range(idx - 1, -1, -1):
        prev = messages[j]
        if isinstance(prev, ToolCall) and prev.id == call_id:
            return prev.name
    # No matching ToolCall in history. Use the call_id as a last-resort
    # placeholder so the request doesn't outright fail validation.
    return call_id or "unknown_tool"


async def _translate_google_stream(stream, model: str = "") -> AsyncIterator[ProviderEvent]:
    """Convert Google's GenerateContentResponse chunks to normalized events.

    Google yields one chunk per content delta. Each chunk's
    `candidates[0].content.parts` may contain `text` and/or `function_call`
    parts; we emit a TextDelta for each text part and a ToolCallEvent for
    each fully-formed function_call. `finish_reason` lands on the last chunk.
    `usage_metadata` is cumulative — we keep the last one and emit it as a
    TurnUsageEvent at the end.
    """

    text_buf = ""
    stop_reason: str | None = None
    call_counter = 0
    input_tokens = 0
    output_tokens = 0
    cached_input_tokens = 0
    cache_write_tokens = 0  # Google reports reads only; writes stay 0
    async for chunk in stream:
        um = getattr(chunk, "usage_metadata", None)
        if um is not None:
            input_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
            cached_input_tokens = int(
                getattr(um, "cached_content_token_count", 0) or 0
            )
        candidates = getattr(chunk, "candidates", None) or []
        if not candidates:
            continue
        cand = candidates[0]
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        for part in parts or []:
            text = getattr(part, "text", None)
            if text:
                text_buf += text
                yield TextDelta(text=text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                call_counter += 1
                fc_name = getattr(fc, "name", "") or ""
                fc_args = getattr(fc, "args", None) or {}
                fc_id = getattr(fc, "id", None) or f"gemini_{call_counter}"
                # Gemini 2.5+ thinking models include a per-Part signature
                # that we must preserve and echo back when this call reappears
                # in conversation history. See:
                # https://ai.google.dev/gemini-api/docs/thought-signatures
                signature = getattr(part, "thought_signature", None)
                yield ToolCallEvent(
                    id=fc_id,
                    name=fc_name,
                    input=dict(fc_args),
                    thought_signature=signature,
                )
        fr = getattr(cand, "finish_reason", None)
        if fr is not None:
            stop_reason = _finish_reason_to_str(fr)
    yield TurnUsageEvent(
        input_tokens=input_tokens, output_tokens=output_tokens, model=model,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    yield AssistantTurnEnd(text=text_buf, stop_reason=stop_reason)


def _finish_reason_to_str(fr: Any) -> str | None:
    if fr is None:
        return None
    # Google's FinishReason is an enum; .value gives the string.
    val = getattr(fr, "value", None)
    if val is None:
        val = str(fr)
    val = str(val).lower()
    if val in ("finish_reason_unspecified", "unspecified", "none"):
        return None
    return val
