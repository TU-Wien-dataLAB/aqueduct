import json

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter
import openai

from gateway.router import get_router
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from management.models import Request
from .decorators import (
    token_authenticated,
    check_limits,
    parse_body,
    ensure_usage,
    log_request,
    check_model_availability,
    catch_router_exceptions,
    process_file_content, tos_accepted
)
from .utils import _get_token_usage, _openai_stream


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.chat.CompletionCreateParams))
@process_file_content
@ensure_usage
@log_request
@check_model_availability
@catch_router_exceptions
async def chat_completions(
        request: ASGIRequest,
        pydantic_model: openai.types.chat.CompletionCreateParams,
        request_log: Request,
        *args,
        **kwargs,
):
    router = get_router()
    chat_completion: CustomStreamWrapper | ModelResponse = await router.acompletion(
        **pydantic_model
    )
    if isinstance(chat_completion, CustomStreamWrapper):
        return StreamingHttpResponse(
            streaming_content=_openai_stream(stream=chat_completion, request_log=request_log),
            content_type='text/event-stream',
        )
    elif isinstance(chat_completion, ModelResponse):
        data = chat_completion.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _get_token_usage(data)
        return JsonResponse(data=data, status=200)
    else:
        raise NotImplementedError(
            f"Completion for response type {type(chat_completion)} is not implemented."
        )
