"""Model provider helpers for OpenAI-compatible clients."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import openai


OPENAI_API_PROVIDER = "openai_api"
OPENCLAW_GATEWAY_PROVIDER = "openclaw_gateway"
OPENAI_PROVIDER_ENV = "AI_SCIENTIST_OPENAI_PROVIDER"

OPENCLAW_BASE_URL_ENV = "AI_SCIENTIST_OPENCLAW_BASE_URL"
OPENCLAW_GATEWAY_TOKEN_ENV = "OPENCLAW_GATEWAY_TOKEN"
OPENCLAW_API_KEY_ENV = "AI_SCIENTIST_OPENCLAW_API_KEY"
OPENCLAW_AGENT_MODEL_ENV = "AI_SCIENTIST_OPENCLAW_AGENT_MODEL"
OPENCLAW_USE_MODEL_HEADER_ENV = "AI_SCIENTIST_OPENCLAW_USE_MODEL_HEADER"

DEFAULT_OPENCLAW_BASE_URL = "http://127.0.0.1:18789/v1"
DEFAULT_OPENCLAW_AGENT_MODEL = "openclaw/default"
SUPPORTED_OPENAI_PROVIDERS = (OPENAI_API_PROVIDER, OPENCLAW_GATEWAY_PROVIDER)


def resolve_openai_provider(provider: str | None = None) -> str:
    """Resolve the selected provider for OpenAI-compatible model calls."""

    resolved = provider or os.environ.get(OPENAI_PROVIDER_ENV, OPENAI_API_PROVIDER)
    if resolved not in SUPPORTED_OPENAI_PROVIDERS:
        raise ValueError(
            f"Unsupported OpenAI provider '{resolved}'. "
            f"Supported providers: {', '.join(SUPPORTED_OPENAI_PROVIDERS)}"
        )
    return resolved


def configure_openai_provider(
    *,
    provider: str | None = None,
    openclaw_base_url: str | None = None,
) -> None:
    """Persist CLI provider choices in env vars used by nested pipeline calls."""

    if provider is not None:
        resolve_openai_provider(provider)
        os.environ[OPENAI_PROVIDER_ENV] = provider
    if openclaw_base_url is not None:
        os.environ[OPENCLAW_BASE_URL_ENV] = normalize_openclaw_base_url(
            openclaw_base_url
        )


def create_openai_client(
    provider: str | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    max_retries: int | None = None,
    **kwargs: Any,
) -> Any:
    """Create an OpenAI-compatible client for the selected provider.

    ``openai_api`` preserves the repository's original OpenAI SDK behavior and
    uses ``OPENAI_API_KEY`` through the SDK when ``api_key`` is omitted.

    ``openclaw_gateway`` talks to a local OpenClaw Gateway OpenAI-compatible
    endpoint. OpenClaw owns ChatGPT/Codex OAuth login and token refresh; this
    repository only sends normal model requests to the gateway.
    """

    resolved_provider = resolve_openai_provider(provider)
    client_kwargs: dict[str, Any] = dict(kwargs)
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries

    if resolved_provider == OPENCLAW_GATEWAY_PROVIDER:
        client_kwargs["api_key"] = _resolve_openclaw_api_key(api_key)
        client_kwargs["base_url"] = normalize_openclaw_base_url(
            base_url or os.environ.get(OPENCLAW_BASE_URL_ENV, DEFAULT_OPENCLAW_BASE_URL)
        )
        client = openai.OpenAI(**client_kwargs)
        return OpenClawGatewayClient(client)

    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    return openai.OpenAI(**client_kwargs)


def normalize_openclaw_base_url(base_url: str) -> str:
    """Ensure OpenAI SDK base_url points at OpenClaw's /v1 surface."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _resolve_openclaw_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    return (
        os.environ.get(OPENCLAW_API_KEY_ENV)
        or os.environ.get(OPENCLAW_GATEWAY_TOKEN_ENV)
        or "not-needed"
    )


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class OpenClawGatewayClient:
    """Thin wrapper around OpenAI SDK for OpenClaw Gateway quirks.

    OpenClaw's OpenAI-compatible endpoints are agent-first. By default, requests
    are sent to ``openclaw/default`` while the requested backend model is copied
    into ``x-openclaw-model``. Set ``AI_SCIENTIST_OPENCLAW_USE_MODEL_HEADER=0``
    to pass model ids through unchanged.
    """

    def __init__(self, client: openai.OpenAI):
        self._client = client
        self.chat = SimpleNamespace(
            completions=_OpenClawChatCompletions(client.chat.completions)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _OpenClawChatCompletions:
    def __init__(self, completions: Any):
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        if _env_flag(OPENCLAW_USE_MODEL_HEADER_ENV, True):
            requested_model = kwargs.get("model")
            if requested_model:
                extra_headers = dict(kwargs.pop("extra_headers", {}) or {})
                header_names = {name.lower() for name in extra_headers}
                if "x-openclaw-model" not in header_names:
                    extra_headers["x-openclaw-model"] = requested_model
                kwargs["extra_headers"] = extra_headers
                kwargs["model"] = os.environ.get(
                    OPENCLAW_AGENT_MODEL_ENV, DEFAULT_OPENCLAW_AGENT_MODEL
                )
        return self._completions.create(**kwargs)
