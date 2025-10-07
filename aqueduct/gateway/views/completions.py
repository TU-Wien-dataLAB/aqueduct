import json

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter
import openai

from gateway.config import get_router
from litellm import TextCompletionStreamWrapper
from litellm.types.utils import TextCompletionResponse

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
from .utils import _usage_from_bytes, _openai_stream


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.CompletionCreateParams))
@ensure_usage
@log_request
@check_model_availability
@catch_router_exceptions
async def completions(
        request: ASGIRequest,
        pydantic_model: openai.types.CompletionCreateParams,
        request_log: Request,
        *args,
        **kwargs,
):
    router = get_router()
    completion: TextCompletionResponse | TextCompletionStreamWrapper = await router.atext_completion(
        **pydantic_model
    )
    if isinstance(completion, TextCompletionStreamWrapper):
        return StreamingHttpResponse(
            streaming_content=_openai_stream(stream=completion, request_log=request_log),
            headers={'Content-Type': 'text/event-stream'},
        )
    elif isinstance(completion, TextCompletionResponse):
        data = completion.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _usage_from_bytes(json.dumps(data).encode("utf-8"))
        return JsonResponse(data=data, status=200)
    else:
        raise NotImplementedError(
            f"Completion for response type {type(completion)} is not implemented."
        )
