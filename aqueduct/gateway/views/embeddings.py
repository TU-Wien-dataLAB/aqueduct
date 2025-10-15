import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm.types.utils import EmbeddingResponse
from management.models import Request
from pydantic import TypeAdapter

from gateway.config import get_router

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    ensure_usage,
    log_request,
    parse_body,
    token_authenticated,
    tos_accepted,
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
