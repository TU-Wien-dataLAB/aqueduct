from typing import Any

import openai
from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from pydantic import TypeAdapter

from gateway.config import get_router
from management.models import Request

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    ensure_usage,
    log_request,
    normalize_reasoning_fields,
    parse_body,
    process_file_content,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from .utils import RawJsonResponse, RawStreamingResponse, get_token_usage


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(openai.types.chat.CompletionCreateParams))
@check_limits
@process_file_content
@ensure_usage
@resolve_alias
@check_model_availability
@normalize_reasoning_fields
@log_request
@catch_router_exceptions
async def chat_completions(
    request: ASGIRequest,
    pydantic_model: openai.types.chat.CompletionCreateParams,
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse | RawStreamingResponse:
    router = get_router()
    chat_completion: CustomStreamWrapper | ModelResponse = await router.acompletion(
        **pydantic_model
    )
    if isinstance(chat_completion, CustomStreamWrapper):
        return RawStreamingResponse(streaming_content=chat_completion, request_log=request_log)
    if isinstance(chat_completion, ModelResponse):
        request_log.token_usage = get_token_usage(chat_completion)
        return RawJsonResponse(data=chat_completion, status=200)
    raise NotImplementedError(
        f"Completion for response type {type(chat_completion)} is not implemented."
    )
