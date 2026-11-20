import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import openai
import yaml
from django.conf import settings
from litellm import Router

log = logging.getLogger("aqueduct")

_openai_client_cache: dict[
    tuple[str, int], tuple[asyncio.AbstractEventLoop, openai.AsyncClient]
] = {}
_files_api_client_cache: dict[int, tuple[asyncio.AbstractEventLoop, openai.AsyncClient]] = {}


def get_files_api_client() -> openai.AsyncClient:
    """
    Get configured OpenAI client for Files/Batches API.

    Uses the OpenAI Python SDK which provides:
    - Proper response models (FileObject, Batch, etc.)
    - Automatic error handling and retries
    - Type safety

    The client is cached per event loop.
    """
    if not settings.AQUEDUCT_FILES_API_URL:
        raise ValueError("AQUEDUCT_FILES_API_URL must be configured")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _build_files_api_client()

    cached = _files_api_client_cache.get(id(loop))
    if cached is not None and not cached[0].is_closed():
        return cached[1]

    client = _build_files_api_client()
    _files_api_client_cache[id(loop)] = (loop, client)
    _evict_closed_loop_clients(_files_api_client_cache)
    return client


def _build_files_api_client() -> openai.AsyncClient:
    if not settings.AQUEDUCT_FILES_API_URL:
        raise ValueError("AQUEDUCT_FILES_API_URL must be configured")
    return openai.AsyncClient(
        base_url=f"{settings.AQUEDUCT_FILES_API_URL.rstrip('/')}/v1",
        api_key=settings.AQUEDUCT_FILES_API_KEY or "unused",
        timeout=60.0,
    )


def _validate_router_config(config: dict[str, Any]) -> None:
    # RPM belongs to the public model, not to individual SDK deployments.
    limits: dict[str, int | None] = {}
    for model in config.get("model_list", []):
        rpm = model.get("model_info", {}).get("rpm")
        if rpm is not None and (type(rpm) is not int or rpm <= 0):
            raise RuntimeError("model_info.rpm must be a positive integer")
        name = model.get("model_name")
        if name in limits and limits[name] != rpm:
            raise RuntimeError(f"Conflicting model_info.rpm values for '{name}'")
        limits[name] = rpm

    sections = [config, config.get("default_litellm_params") or {}]
    for model in config.get("model_list", []):
        sections.extend([model, model.get("litellm_params") or {}])
    if any(section.get("rpm") is not None for section in sections):
        raise RuntimeError("Move router/deployment RPM limits to model_info.rpm")

    # Validate alias uniqueness
    alias_to_model: dict[str, str] = {}
    model_list = config.get("model_list", [])
    for model in model_list:
        model_name = model.get("model_name")
        aliases = model.get("model_info", {}).get("aliases", [])
        for alias in aliases:
            if alias in alias_to_model:
                raise RuntimeError(
                    f"Duplicate alias '{alias}' found in router config. "
                    f"Alias is used by both '{alias_to_model[alias]}' and '{model_name}'."
                )
            alias_to_model[alias] = model_name


@lru_cache(maxsize=1)
def get_router_config() -> dict[str, Any]:
    path = settings.LITELLM_ROUTER_CONFIG_FILE_PATH
    if path is None:
        raise RuntimeError("LITELLM_ROUTER_CONFIG_FILE_PATH must be configured")
    try:
        log.info("Loading router config from %s", path)
        with Path(path).open() as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise RuntimeError(f"Unable to load router config from {path}") from None

    if data is None:
        raise RuntimeError(f"Unable to load router config from {path}")

    _validate_router_config(data)

    return cast("dict[str, Any]", data)


@lru_cache(maxsize=1)
def model_entries() -> dict[str, dict[str, Any]]:
    # LiteLLM balances load over entries that repeat a `model_name`; the first one describes it.
    entries: dict[str, dict[str, Any]] = {}
    for entry in get_router_config().get("model_list") or ():
        entries.setdefault(entry["model_name"], entry)
    return entries


@lru_cache(maxsize=1)
def model_names() -> tuple[str, ...]:
    return tuple(model_entries())


def model_info_with_context(model: Mapping[str, Any]) -> dict[str, Any]:
    model_info = dict(model.get("model_info") or {})
    if "max_input_tokens" not in model_info:
        max_tokens = model_info.get("max_tokens")
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            model_info["max_input_tokens"] = max_tokens
    return model_info


@lru_cache(maxsize=1)
def get_router() -> Router:
    config = get_router_config()
    if config is None:
        raise RuntimeError("Router config not found!")
    return Router(**config, num_retries=settings.API_MAX_RETRIES)


def get_openai_client(model: str) -> openai.AsyncClient:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _build_openai_client(model)

    key = (model, id(loop))
    cached = _openai_client_cache.get(key)
    if cached is not None and not cached[0].is_closed():
        return cached[1]

    client = _build_openai_client(model)
    _openai_client_cache[key] = (loop, client)
    _evict_closed_loop_clients(_openai_client_cache)
    return client


def _build_openai_client(model: str) -> openai.AsyncClient:
    router = get_router()
    deployment = router.get_deployment(model_id=model)
    if deployment is None:
        raise ValueError(f"Deployment for model '{model}' not found!")
    litellm_params = deployment.litellm_params

    timeout = litellm_params.timeout
    if isinstance(timeout, str):
        # litellm passes env vars as "os.environ/VAR_NAME" strings
        timeout = float(os.environ[timeout.removeprefix("os.environ/")])

    return openai.AsyncClient(
        api_key=litellm_params.api_key,
        base_url=litellm_params.api_base,
        timeout=timeout,
        max_retries=settings.API_MAX_RETRIES,
    )


def _evict_closed_loop_clients(
    cache: dict[Any, tuple[asyncio.AbstractEventLoop, openai.AsyncClient]],
) -> None:
    stale = [key for key, (loop, _) in cache.items() if loop.is_closed()]
    for key in stale:
        del cache[key]


class MCPServerConfig(TypedDict):
    type: Literal["streamable-http"]  # only support "streamable-http"
    url: str
    description: NotRequired[str]
    tags: NotRequired[list[str]]
    icon_url: NotRequired[str]


@dataclass(frozen=True)
class ResolvedModel:
    accounting_name: str | None
    routing_name: str


def resolve_model(name: str) -> ResolvedModel:
    """Resolve accounting identity without changing deployment-specific routing."""
    model_list = get_router_config().get("model_list", [])

    # Public names and aliases take precedence over explicit deployment IDs.
    for model in model_list:
        model_name = model.get("model_name")
        aliases = model.get("model_info", {}).get("aliases", [])
        if isinstance(model_name, str) and (model_name == name or name in aliases):
            return ResolvedModel(model_name, model_name)

    for model in model_list:
        model_name = model.get("model_name")
        deployment_id = model.get("model_info", {}).get("id")
        if isinstance(model_name, str) and deployment_id is not None and str(deployment_id) == name:
            return ResolvedModel(model_name, name)

    return ResolvedModel(None, name)


def get_model_request_limit_multiplier(model_name: str) -> float:
    """
    Get the request limit multiplier for a given model.
    Returns 1.0 if no multiplier is configured.

    Args:
        model_name: The model name (or alias, will be resolved)

    Returns:
        The request limit multiplier (defaults to 1.0)
    """
    resolved = resolve_model(model_name).routing_name
    config = get_router_config()
    model_list = config.get("model_list", [])

    for model in model_list:
        if model.get("model_name") == resolved:
            multiplier = model.get("model_info", {}).get("request_limit_multiplier")
            if multiplier is not None:
                return float(multiplier)
            break

    return 1.0


@lru_cache(maxsize=1)
def get_all_model_request_limit_multipliers() -> dict[str, float]:
    """
    Get request limit multipliers for all configured models.

    Returns:
        Dict mapping model names to their multipliers (defaults to 1.0)
    """
    config = get_router_config()
    model_list = config.get("model_list", [])
    multipliers: dict[str, float] = {}

    for model in model_list:
        model_name = model.get("model_name")
        if model_name:
            multiplier = model.get("model_info", {}).get("request_limit_multiplier")
            multipliers[model_name] = float(multiplier) if multiplier is not None else 1.0

    return multipliers


def get_model_rpm(model_name: str) -> int | None:
    """Get the public model's RPM limit from ``model_info.rpm``, if configured."""
    resolved = resolve_model(model_name).accounting_name
    for model in get_router_config().get("model_list", []):
        if resolved is not None and model.get("model_name") == resolved:
            return cast("int | None", model.get("model_info", {}).get("rpm"))
    return None


def get_model_max_parallel_requests(model_name: str) -> int | None:
    """
    Get the max parallel requests (concurrency limit) for a given model from the
    router config's ``model_info.max_parallel_requests``.

    Mirrors LiteLLM's ``max_parallel_requests`` setting, and falls back to the
    ``default_max_parallel_requests`` router setting. Returns ``None`` if neither
    is configured.

    Args:
        model_name: A public model name, alias, or configured deployment ID.

    Returns:
        The configured max parallel requests, or ``None`` if not set.
    """
    resolved = resolve_model(model_name).accounting_name
    config = get_router_config()
    model_list = config.get("model_list", [])

    for model in model_list:
        if resolved is not None and model.get("model_name") == resolved:
            max_parallel_requests = model.get("model_info", {}).get("max_parallel_requests")
            if max_parallel_requests is not None:
                return int(max_parallel_requests)
            if (default := config.get("default_max_parallel_requests")) is not None:
                return int(default)
            break

    return None


def get_model_supports_response_format(model_name: str) -> bool:
    """
    Whether the model's upstream API accepts the `response_format` request parameter.

    Returns False only when the model's model_info explicitly sets
    `supports_response_format: false` (e.g. gpt-image models, which only return
    b64_json and reject the parameter). Defaults to True.

    Args:
        model_name: The model name (or alias, will be resolved)

    Returns:
        Whether response_format may be sent to the model
    """
    resolved = resolve_model(model_name).routing_name
    config = get_router_config()
    model_list = config.get("model_list", [])

    for model in model_list:
        if model.get("model_name") == resolved:
            return model.get("model_info", {}).get("supports_response_format", True) is not False

    return True


@lru_cache(maxsize=1)
def get_mcp_config() -> dict[str, MCPServerConfig]:
    path = settings.MCP_CONFIG_FILE_PATH
    try:
        with Path(path).open() as f:
            data = json.load(f)["mcpServers"]
            return {server: cast("MCPServerConfig", config) for server, config in data.items()}
    except (FileNotFoundError, TypeError, json.JSONDecodeError, KeyError):
        raise RuntimeError(f"Unable to load MCP config from {path}") from None
