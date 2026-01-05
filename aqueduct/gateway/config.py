import json
import logging
from functools import lru_cache
from typing import Literal, NotRequired, TypedDict

import openai
import yaml
from django.conf import settings
from litellm import Router

log = logging.getLogger("aqueduct")


def _validate_router_config(config: dict):
    # Validate alias uniqueness
    alias_to_model = {}
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
def get_router_config() -> dict:
    path = settings.LITELLM_ROUTER_CONFIG_FILE_PATH
    try:
        log.info(f"Loading router config from {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
    except (FileNotFoundError, TypeError):
        raise RuntimeError(f"Unable to load router config from {path}")

    _validate_router_config(data)

    return data


@lru_cache(maxsize=1)
def get_router() -> Router:
    config = get_router_config()
    if config is None:
        raise RuntimeError("Router config not found!")
    return Router(**config)


@lru_cache(maxsize=32)
def get_openai_client(model: str) -> openai.AsyncClient:
    router = get_router()
    deployment = router.get_deployment(model_id=model)
    if deployment is None:
        raise ValueError(f"Deployment for model '{model}' not found!")
    litellm_params = deployment.litellm_params

    return openai.AsyncClient(
        api_key=litellm_params.api_key,
        base_url=litellm_params.api_base,
        timeout=litellm_params.timeout,
    )


class MCPServerConfig(TypedDict):
    type: Literal["streamable-http"]  # only support "streamable-http"
    url: str
    description: NotRequired[str]
    tags: NotRequired[list[str]]
    icon_url: NotRequired[str]


def resolve_model_alias(model_or_alias: str) -> str:
    """
    Resolve a model alias to its actual model name.
    If the input is already a model name (not an alias), return it unchanged.

    Args:
        model_or_alias: Either a model alias or a model name

    Returns:
        The actual model name
    """
    config = get_router_config()
    model_list = config.get("model_list", [])

    # Build alias - model mapping
    for model in model_list:
        model_name = model.get("model_name")
        if model_name == model_or_alias:
            return model_name

        aliases = model.get("model_info", {}).get("aliases", [])
        if model_or_alias in aliases:
            return model_name

    # Not an alias or a model; still return as-is to return standard error down the line
    return model_or_alias


@lru_cache(maxsize=1)
def get_mcp_config() -> dict[str, MCPServerConfig]:
    path = settings.MCP_CONFIG_FILE_PATH
    try:
        with open(path) as f:
            data = json.load(f)["mcpServers"]
            return {server: MCPServerConfig(**config) for server, config in data.items()}
    except (FileNotFoundError, TypeError, json.JSONDecodeError, KeyError):
        raise RuntimeError(f"Unable to load MCP config from {path}")
