import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm import TextCompletionStreamWrapper
from litellm.types.utils import TextCompletionResponse
from pydantic import TypeAdapter

from gateway.config import get_router
from management.models import Request

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    ensure_usage,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from .utils import _get_token_usage, _openai_stream


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.CompletionCreateParams))
@ensure_usage
@log_request
@resolve_alias
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
    completion: (
        TextCompletionResponse | TextCompletionStreamWrapper
    ) = await router.atext_completion(**pydantic_model)
    if isinstance(completion, TextCompletionStreamWrapper):
        return StreamingHttpResponse(
            streaming_content=_openai_stream(stream=completion, request_log=request_log),
            headers={"Content-Type": "text/event-stream"},
        )
    elif isinstance(completion, TextCompletionResponse):
        data = completion.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _get_token_usage(data)
        return JsonResponse(data=data, status=200)
    else:
        raise NotImplementedError(
            f"Completion for response type {type(completion)} is not implemented."
        )
