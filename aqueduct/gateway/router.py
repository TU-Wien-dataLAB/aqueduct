from functools import lru_cache

import openai
import yaml
from django.conf import settings
from litellm import Router


@lru_cache(maxsize=1)
def get_router_config() -> dict:
    path = settings.LITELLM_ROUTER_CONFIG_FILE_PATH
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except (FileNotFoundError, TypeError):
        raise RuntimeError(f'Unable to load router config from {path}')
    return data


@lru_cache(maxsize=1)
def get_router() -> Router:
    config = get_router_config()
    if config is None:
        raise RuntimeError(f"Router config not found!")
    return Router(**config)


@lru_cache(maxsize=32)
def get_openai_client(model: str) -> openai.AsyncClient:
    router = get_router()
    deployment = router.get_deployment(model_id=model)
    if deployment is None:
        raise ValueError(f"Deployment for model '{model}' not found!")
    litellm_params = deployment.litellm_params

    return openai.AsyncClient(api_key=litellm_params.api_key, base_url=litellm_params.api_base, timeout=litellm_params.timeout)
