from functools import lru_cache

import yaml
from django.conf import settings
from litellm import Router
from litellm.types.router import RouterConfig


@lru_cache(maxsize=1)
def get_router_config() -> RouterConfig:
    path = settings.LITELLM_ROUTER_CONFIG_FILE_PATH
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (FileNotFoundError, TypeError):
        raise RuntimeError(f'Unable to load router config from {path}')
    return RouterConfig.model_validate(data)


@lru_cache(maxsize=1)
def get_router() -> Router:
    config = get_router_config()
    if config is None:
        raise RuntimeError(f"Router config not found!")
    return Router(**config.model_dump())
