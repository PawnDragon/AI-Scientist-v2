"""Model provider helpers for OpenAI-compatible clients."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import openai


OPENAI_API_PROVIDER = "openai_api"
OPENCLAW_GATEWAY_PROVIDER = "openclaw_gateway"
OPENAI_PROVIDER_ENV = "AI_SCIENTIST_OPENAI_PROVIDER"
REASONING_EFFORT_ENV = "AI_SCIENTIST_REASONING_EFFORT"
REASONING_MAX_COMPLETION_TOKENS_ENV = (
    "AI_SCIENTIST_REASONING_MAX_COMPLETION_TOKENS"
)

OPENCLAW_BASE_URL_ENV = "AI_SCIENTIST_OPENCLAW_BASE_URL"
OPENCLAW_GATEWAY_TOKEN_ENV = "OPENCLAW_GATEWAY_TOKEN"
OPENCLAW_API_KEY_ENV = "AI_SCIENTIST_OPENCLAW_API_KEY"
OPENCLAW_AGENT_MODEL_ENV = "AI_SCIENTIST_OPENCLAW_AGENT_MODEL"
OPENCLAW_USE_MODEL_HEADER_ENV = "AI_SCIENTIST_OPENCLAW_USE_MODEL_HEADER"

DEFAULT_OPENCLAW_BASE_URL = "http://127.0.0.1:18789/v1"
DEFAULT_OPENCLAW_AGENT_MODEL = "openclaw/default"
DEFAULT_XHIGH_MIN_MAX_COMPLETION_TOKENS = 12000
SUPPORTED_OPENAI_PROVIDERS = (OPENAI_API_PROVIDER, OPENCLAW_GATEWAY_PROVIDER)
SUPPORTED_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


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
    reasoning_effort: str | None = None,
) -> None:
    """Persist CLI provider choices in env vars used by nested pipeline calls."""

    if provider is not None:
        resolve_openai_provider(provider)
        os.environ[OPENAI_PROVIDER_ENV] = provider
    if openclaw_base_url is not None:
        os.environ[OPENCLAW_BASE_URL_ENV] = normalize_openclaw_base_url(
            openclaw_base_url
        )
    if reasoning_effort is not None:
        os.environ[REASONING_EFFORT_ENV] = validate_reasoning_effort(
            reasoning_effort
        )


def validate_reasoning_effort(reasoning_effort: str) -> str:
    normalized = reasoning_effort.strip().lower()
    if normalized not in SUPPORTED_REASONING_EFFORTS:
        raise ValueError(
            f"Unsupported reasoning effort '{reasoning_effort}'. "
            f"Supported values: {', '.join(SUPPORTED_REASONING_EFFORTS)}"
        )
    return normalized


def get_configured_reasoning_effort(default: str | None = None) -> str | None:
    configured = os.environ.get(REASONING_EFFORT_ENV)
    if configured is None or configured == "":
        return default
    return validate_reasoning_effort(configured)


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
    return OpenAIAPIClient(openai.OpenAI(**client_kwargs))


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


def _supports_reasoning_effort(model: str | None) -> bool:
    if not model:
        return False
    model_id = model.split("/")[-1].lower()
    return (
        model_id.startswith(("o1", "o3", "o4"))
        or model_id.startswith("gpt-5")
        or "codex" in model_id
    )


def _requires_max_completion_tokens(model: str | None) -> bool:
    if not model:
        return False
    model_id = model.split("/")[-1].lower()
    return model_id.startswith(("o1", "o3", "o4", "gpt-5")) or "codex" in model_id


def _apply_reasoning_effort(kwargs: dict[str, Any], model: str | None) -> None:
    reasoning_effort = get_configured_reasoning_effort()
    if reasoning_effort is None or not _supports_reasoning_effort(model):
        return
    if kwargs.get("tools") or kwargs.get("tool_choice"):
        return
    kwargs["reasoning_effort"] = reasoning_effort


def _apply_token_limit_parameter(kwargs: dict[str, Any], model: str | None) -> None:
    if not _requires_max_completion_tokens(model):
        return
    if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    if "max_completion_tokens" not in kwargs:
        return

    min_tokens = _get_min_reasoning_max_completion_tokens()
    if min_tokens is not None and kwargs["max_completion_tokens"] < min_tokens:
        kwargs["max_completion_tokens"] = min_tokens


def _get_min_reasoning_max_completion_tokens() -> int | None:
    configured = os.environ.get(REASONING_MAX_COMPLETION_TOKENS_ENV)
    if configured:
        try:
            return max(int(configured), 1)
        except ValueError as exc:
            raise ValueError(
                f"{REASONING_MAX_COMPLETION_TOKENS_ENV} must be an integer, "
                f"got {configured!r}."
            ) from exc

    if get_configured_reasoning_effort() == "xhigh":
        return DEFAULT_XHIGH_MIN_MAX_COMPLETION_TOKENS
    return None


def _apply_sampling_parameters(kwargs: dict[str, Any], model: str | None) -> None:
    if not _requires_max_completion_tokens(model):
        return
    if kwargs.get("temperature") not in (None, 1, 1.0):
        kwargs["temperature"] = 1


class OpenAIAPIClient:
    """OpenAI SDK wrapper that applies AI-Scientist runtime defaults."""

    def __init__(self, client: openai.OpenAI):
        self._client = client
        self.chat = SimpleNamespace(
            completions=_ReasoningChatCompletions(client.chat.completions)
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ReasoningChatCompletions:
    def __init__(self, completions: Any):
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model")
        _apply_reasoning_effort(kwargs, model)
        _apply_token_limit_parameter(kwargs, model)
        _apply_sampling_parameters(kwargs, model)
        return self._completions.create(**kwargs)


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
        requested_model = kwargs.get("model")
        _apply_reasoning_effort(kwargs, requested_model)
        _apply_token_limit_parameter(kwargs, requested_model)
        _apply_sampling_parameters(kwargs, requested_model)
        if _env_flag(OPENCLAW_USE_MODEL_HEADER_ENV, True):
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
