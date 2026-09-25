from typing import Any

from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from gateway.config import get_router_config
from management.models import Request, Token

from .decorators import log_request, token_authenticated, tos_accepted

MODEL_CREATION_TIMESTAMP = int(timezone.now().timestamp())

def _model_group_info_entry(model: dict[str, Any]) -> dict[str, Any]:
    """Build a LiteLLM ``/model_group/info`` entry for a router config model.

    The generated router config only carries ``max_tokens`` (the context
    length); LiteLLM-aware clients read ``max_input_tokens`` as the context
    window, so derive it here — in this HTTP response only, never in the
    config itself (a config-level value would re-trigger the Router's
    tiktoken pre-checks).
    """
    model_info = dict(model.get("model_info") or {})
    if "max_input_tokens" not in model_info:
        max_tokens = model_info.get("max_tokens")
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            model_info["max_input_tokens"] = max_tokens
    return {
        "model_group": model["model_name"],
        "model_name": model["model_name"],
        "model_info": model_info,
    }


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def models(
    request: ASGIRequest, token: Token, request_log: Request, *args: Any, **kwargs: Any
) -> JsonResponse:
    router_config = get_router_config()
    model_list: list[dict[str, Any]] = router_config["model_list"]
    excluded_models = set(await sync_to_async(token.model_exclusion_list)())

    return JsonResponse(
        data={
            "data": [
                {
                    "id": model["model_name"],
                    "object": "model",
                    "created": MODEL_CREATION_TIMESTAMP,
                    "owned_by": "aqueduct",
                }
                for model in model_list
                if model["model_name"] not in excluded_models
            ],
            "object": "list",
        }
    )

@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def model_group_info(
    request: ASGIRequest, token: Token, request_log: Request, *args: Any, **kwargs: Any
) -> JsonResponse:
    """LiteLLM-style rich model metadata endpoint.

    Returns one entry per configured model (JSON array) so that LiteLLM-aware
    clients — which probe /model_group/info before falling back to
    /v1/models — read token limits and capabilities directly from the gateway
    instead of guessing from bundled model catalogs.
    """
    router_config = get_router_config()
    model_list: list[dict[str, Any]] = router_config["model_list"]
    excluded_models = set(await sync_to_async(token.model_exclusion_list)())

    return JsonResponse(
        [
            _model_group_info_entry(model)
            for model in model_list
            if model["model_name"] not in excluded_models
        ],
        safe=False,
    )
