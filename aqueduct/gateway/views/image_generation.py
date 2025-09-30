import json

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm.types.utils import ImageResponse
from pydantic import TypeAdapter
import openai

from gateway.router import get_router
from management.models import Request
from .decorators import (
    token_authenticated,
    check_limits,
    parse_body,
    log_request,
    check_model_availability,
    catch_router_exceptions,
)
from .utils import _get_token_usage


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@check_limits
@parse_body(model=TypeAdapter(openai.types.ImageGenerateParams))
@log_request
@check_model_availability
@catch_router_exceptions
async def image_generation(
        request: ASGIRequest,
        pydantic_model: openai.types.ImageGenerateParams,
        request_log: Request,
        *args,
        **kwargs,
):
    router = get_router()
    resp: ImageResponse = router.image_generation(**pydantic_model)
    data = resp.model_dump(exclude_unset=True)
    request_log.token_usage = _get_token_usage(data)

    return JsonResponse(data)
