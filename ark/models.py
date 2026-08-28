"""Known model context-window sizes.

Hand-maintained — providers don't expose this reliably via API. Used purely
for the per-turn usage indicator (UI feedback to the user / agent). When
the model is missing or has an extension flag (Claude's 1M beta, etc.), the
indicator falls back to raw token counts without a percentage.

Agents can override via `agents.<name>.max_context_tokens` in config.
"""

from __future__ import annotations

# Provider-native model id → max input tokens.
KNOWN_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic — native 200k; 1M is available as a beta header but we don't
    # opt into it from Ark, so 200k is the operative ceiling.
    "claude-opus-4-7": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    # OpenAI
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5.4-mini": 400_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    # Google (Gemini 2.5 family — 1M input, 2M available with extension)
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    # Moonshot (also reachable via OpenRouter's moonshotai/ namespace,
    # which context_window_for's bare-name fallback resolves).
    "kimi-k3": 256_000,
}


def context_window_for(model: str, override: int | None = None) -> int | None:
    """Return the input-token ceiling for a model.

    `override` (per-agent config) wins. Otherwise look up in the table.
    Returns None if neither yields a value — clients should show raw counts
    instead of a percentage.
    """

    if override is not None and override > 0:
        return int(override)
    # OpenRouter model ids look like `<vendor>/<model>` — strip the prefix
    # for the lookup so e.g. `anthropic/claude-sonnet-4-6` resolves.
    bare = model.rsplit("/", 1)[-1] if "/" in model else model
    return KNOWN_CONTEXT_WINDOWS.get(bare)
