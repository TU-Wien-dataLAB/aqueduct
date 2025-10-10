import json

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter
import openai

from gateway.router import get_router
from litellm.types.utils import EmbeddingResponse

from management.models import Request
from .decorators import (
    token_authenticated,
    check_limits,
    parse_body,
    ensure_usage,
    log_request,
    check_model_availability,
    catch_router_exceptions, tos_accepted,
)
from .utils import _get_token_usage


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.EmbeddingCreateParams))
@ensure_usage
@log_request
@check_model_availability
@catch_router_exceptions
async def embeddings(
        request: ASGIRequest,
        pydantic_model: openai.types.EmbeddingCreateParams,
        request_log: Request,
        *args,
        **kwargs,
):
    router = get_router()
    embedding: EmbeddingResponse = await router.aembedding(**pydantic_model)
    data = embedding.model_dump(exclude_none=True, exclude_unset=True)
    request_log.token_usage = _get_token_usage(data)
    return JsonResponse(data=data, status=200)
