from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET

from gateway.config import get_router_config
from management.models import Token

from .decorators import log_request, token_authenticated, tos_accepted

MODEL_CREATION_TIMESTAMP = int(timezone.now().timestamp())


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
async def models(request: ASGIRequest, token: Token, *args, **kwargs):
    router_config = get_router_config()
    model_list: list[dict] = router_config["model_list"]
    excluded_models = set(await sync_to_async(token.model_exclusion_list)())

    return JsonResponse(
        data=dict(
            data=[
                {
                    "id": model["model_name"],
                    "object": "model",
                    "created": MODEL_CREATION_TIMESTAMP,
                    "owned_by": "aqueduct",
                }
                for model in model_list
                if model["model_name"] not in excluded_models
            ],
            object="list",
        )
    )
