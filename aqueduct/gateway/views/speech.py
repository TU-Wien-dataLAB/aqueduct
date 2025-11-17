import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from openai import HttpxBinaryResponseContent
from pydantic import TypeAdapter

from gateway.config import get_router

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


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.audio.SpeechCreateParams))
@resolve_alias
@ensure_usage
@log_request
@check_model_availability
@catch_router_exceptions
async def speech(
    request: ASGIRequest, pydantic_model: openai.types.audio.SpeechCreateParams, *args, **kwargs
):
    router = get_router()

    speech_output: HttpxBinaryResponseContent = await router.aspeech(**pydantic_model)
    byte_stream = await speech_output.aiter_bytes()
    return StreamingHttpResponse(streaming_content=byte_stream, content_type="text/event-stream")
