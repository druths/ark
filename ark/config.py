"""Load and validate ~/.ark/config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths


class ConfigError(Exception):
    pass


KNOWN_PROVIDER_TYPES = frozenset({"anthropic", "google", "openai", "openrouter"})


@dataclass
class ServerConfig:
    host: str
    port: int
    auth_secret: str


@dataclass
class ProviderConfig:
    provider_type: str  # one of KNOWN_PROVIDER_TYPES
    api_key: str
    base_url: str | None = None


@dataclass
class MCPServerConfig:
    name: str
    transport: str  # "stdio" | "http"
    # stdio
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # http
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # Applied per tool call (list_tools uses the same deadline).
    timeout_seconds: float = 30.0


@dataclass
class AgentConfig:
    name: str
    provider: str  # references a key in Config.providers
    model: str
    workspace: Path
    always_loaded_skills: list[str] = field(default_factory=list)
    # Override for the model's input-token ceiling. Optional — falls back to
    # the table in ark.models. Used purely to compute the usage indicator.
    max_context_tokens: int | None = None
    # MCP servers this agent may access. Names reference Config.mcp_servers.
    # Servers not listed here are invisible to the agent even if configured
    # globally.
    mcp_servers: list[str] = field(default_factory=list)
    # Subset of mcp_servers whose tools are exposed on every turn without
    # requiring the agent to call load_skill first. Mirrors always_loaded_skills.
    always_loaded_mcp_servers: list[str] = field(default_factory=list)
    # Automatic session compaction — see docs/sessions.md.
    compaction_enabled: bool = True
    compaction_threshold: float = 0.85  # fraction of context_window that triggers
    # Prompt caching (mine-capstone#697): cache_control markers on providers
    # that support them (Anthropic native + anthropic/* via OpenRouter).
    prompt_caching: bool = True
    # In-memory tool-result elision (never touches rows): once the in-view
    # tool-result bytes exceed tool_result_max_bytes, ONE pass elides
    # results older than the last tool_result_keep_turns user turns that
    # are larger than tool_result_elide_over. Hysteresis by design — a
    # per-turn trim would bust the prompt cache every turn.
    tool_result_keep_turns: int = 2
    tool_result_elide_over: int = 2048
    tool_result_max_bytes: int = 65536


@dataclass
class Config:
    server: ServerConfig
    providers: dict[str, ProviderConfig]
    tools: dict[str, dict[str, Any]]
    agents: dict[str, AgentConfig]
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def load(path: Path | None = None) -> Config:
    path = path or paths.config_path()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {path}: {e}") from e
    return _from_dict(raw)


def _from_dict(raw: dict[str, Any]) -> Config:
    server = _server(raw.get("server") or {})
    providers = _providers(raw.get("providers") or {})
    tools = dict(raw.get("tools") or {})
    mcp_servers = _mcp_servers(raw.get("mcp_servers") or {})
    agents = _agents(raw.get("agents") or {}, providers, mcp_servers)
    return Config(
        server=server,
        providers=providers,
        tools=tools,
        agents=agents,
        mcp_servers=mcp_servers,
    )


def _mcp_servers(raw: dict[str, Any]) -> dict[str, MCPServerConfig]:
    out: dict[str, MCPServerConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"mcp_servers.{name} must be an object")
        transport = cfg.get("transport")
        if transport not in ("stdio", "http"):
            raise ConfigError(
                f"mcp_servers.{name}.transport must be 'stdio' or 'http'"
            )
        if transport == "stdio":
            if not cfg.get("command"):
                raise ConfigError(f"mcp_servers.{name}.command is required for stdio")
            args = cfg.get("args") or []
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ConfigError(
                    f"mcp_servers.{name}.args must be a list of strings"
                )
            env = cfg.get("env") or {}
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()
            ):
                raise ConfigError(
                    f"mcp_servers.{name}.env must be an object of string→string"
                )
        else:  # http
            if not cfg.get("url"):
                raise ConfigError(f"mcp_servers.{name}.url is required for http")
            args = []
            env = {}
        headers = cfg.get("headers") or {}
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            raise ConfigError(
                f"mcp_servers.{name}.headers must be an object of string→string"
            )
        timeout = cfg.get("timeout_seconds", 30.0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ConfigError(
                f"mcp_servers.{name}.timeout_seconds must be a positive number"
            )
        out[name] = MCPServerConfig(
            name=name,
            transport=transport,
            command=cfg.get("command"),
            args=list(args),
            env=dict(env),
            url=cfg.get("url"),
            headers=dict(headers),
            timeout_seconds=float(timeout),
        )
    return out


def _server(raw: dict[str, Any]) -> ServerConfig:
    secret = raw.get("auth_secret")
    if not secret or not isinstance(secret, str):
        raise ConfigError("server.auth_secret is required and must be a non-empty string")
    return ServerConfig(
        host=raw.get("host", "127.0.0.1"),
        port=int(raw.get("port", 7777)),
        auth_secret=secret,
    )


def _providers(raw: dict[str, Any]) -> dict[str, ProviderConfig]:
    out: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"providers.{name} must be an object")
        ptype = cfg.get("provider_type")
        if not ptype:
            raise ConfigError(f"providers.{name}.provider_type is required")
        if ptype not in KNOWN_PROVIDER_TYPES:
            raise ConfigError(
                f"providers.{name}.provider_type {ptype!r} is not one of "
                f"{sorted(KNOWN_PROVIDER_TYPES)}"
            )
        if not cfg.get("api_key"):
            raise ConfigError(f"providers.{name}.api_key is required")
        base_url = cfg.get("base_url")
        if base_url is not None and not isinstance(base_url, str):
            raise ConfigError(f"providers.{name}.base_url must be a string")
        out[name] = ProviderConfig(
            provider_type=ptype, api_key=cfg["api_key"], base_url=base_url
        )
    return out


def _agents(
    raw: dict[str, Any],
    providers: dict[str, ProviderConfig],
    mcp_servers: dict[str, MCPServerConfig],
) -> dict[str, AgentConfig]:
    out: dict[str, AgentConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"agents.{name} must be an object")
        for key in ("provider", "model"):
            if not cfg.get(key):
                raise ConfigError(f"agents.{name}.{key} is required")
        provider = cfg["provider"]
        if provider not in providers:
            raise ConfigError(
                f"agents.{name}.provider '{provider}' is not in providers"
            )
        workspace = (
            Path(cfg["workspace"]).expanduser()
            if cfg.get("workspace")
            else paths.default_agent_workspace(name)
        )
        skills = cfg.get("always_loaded_skills") or []
        if not isinstance(skills, list) or not all(isinstance(s, str) for s in skills):
            raise ConfigError(
                f"agents.{name}.always_loaded_skills must be a list of strings"
            )
        max_ctx = cfg.get("max_context_tokens")
        if max_ctx is not None and (not isinstance(max_ctx, int) or max_ctx <= 0):
            raise ConfigError(
                f"agents.{name}.max_context_tokens must be a positive integer if set"
            )
        agent_mcp = cfg.get("mcp_servers") or []
        if not isinstance(agent_mcp, list) or not all(
            isinstance(s, str) for s in agent_mcp
        ):
            raise ConfigError(
                f"agents.{name}.mcp_servers must be a list of strings"
            )
        for s in agent_mcp:
            if s not in mcp_servers:
                raise ConfigError(
                    f"agents.{name}.mcp_servers references unknown server '{s}'"
                )
        always_mcp = cfg.get("always_loaded_mcp_servers") or []
        if not isinstance(always_mcp, list) or not all(
            isinstance(s, str) for s in always_mcp
        ):
            raise ConfigError(
                f"agents.{name}.always_loaded_mcp_servers must be a list of strings"
            )
        for s in always_mcp:
            if s not in agent_mcp:
                raise ConfigError(
                    f"agents.{name}.always_loaded_mcp_servers references '{s}' "
                    f"which is not in agents.{name}.mcp_servers"
                )
        compaction_enabled = cfg.get("compaction_enabled", True)
        if not isinstance(compaction_enabled, bool):
            raise ConfigError(
                f"agents.{name}.compaction_enabled must be a boolean"
            )
        compaction_threshold = cfg.get("compaction_threshold", 0.85)
        if (
            not isinstance(compaction_threshold, (int, float))
            or not (0.0 < compaction_threshold < 1.0)
        ):
            raise ConfigError(
                f"agents.{name}.compaction_threshold must be a number strictly between 0 and 1"
            )
        prompt_caching = cfg.get("prompt_caching", True)
        if not isinstance(prompt_caching, bool):
            raise ConfigError(f"agents.{name}.prompt_caching must be a boolean")

        def _int_knob(key: str, default: int) -> int:
            value = cfg.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ConfigError(f"agents.{name}.{key} must be a non-negative integer")
            return value

        out[name] = AgentConfig(
            name=name,
            provider=provider,
            model=cfg["model"],
            workspace=workspace,
            always_loaded_skills=list(skills),
            max_context_tokens=max_ctx,
            mcp_servers=list(agent_mcp),
            always_loaded_mcp_servers=list(always_mcp),
            compaction_enabled=compaction_enabled,
            compaction_threshold=float(compaction_threshold),
            prompt_caching=prompt_caching,
            tool_result_keep_turns=_int_knob("tool_result_keep_turns", 2),
            tool_result_elide_over=_int_knob("tool_result_elide_over", 2048),
            tool_result_max_bytes=_int_knob("tool_result_max_bytes", 65536),
        )
    return out
