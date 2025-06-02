import warnings
from typing import Optional

import yaml
from litellm import Router
from litellm.types.router import RouterConfig

_config: Optional[RouterConfig] = None
_router: Optional[Router] = None


def load_router(path: str):
    global _config, _router
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (FileNotFoundError, TypeError):
        warnings.warn(f"Router config file {path} not found")
        return
    _config = RouterConfig.model_validate(data)
    _router = Router(**_config.model_dump())


def get_router() -> Router:
    global _router
    return _router


def get_router_config() -> RouterConfig:
    global _config
    return _config
