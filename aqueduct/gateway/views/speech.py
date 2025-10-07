import litellm
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from openai import HttpxBinaryResponseContent
from pydantic import TypeAdapter
import openai

from gateway.config import get_router, get_router_config
from .decorators import (
    token_authenticated,
    check_limits,
    parse_body,
    ensure_usage,
    log_request,
    check_model_availability,
    catch_router_exceptions, tos_accepted,
)


def validate_tts(pydantic_model: dict):
    router_config = get_router_config()
    model_list: list[dict] = router_config["model_list"]
    model_params = filter(lambda m: m["model_name"] == pydantic_model["model"], model_list)
    for model in model_params:
        if model.get("model_info", {}).get("mode", None) != "audio_speech":
            raise litellm.BadRequestError(f"{model['model_name']} does not support text-to-speech.",
                                          model['model_name'], llm_provider=None)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@check_limits
@parse_body(model=TypeAdapter(openai.types.audio.SpeechCreateParams))
@ensure_usage
@log_request
@check_model_availability
@catch_router_exceptions
async def speech(
        request: ASGIRequest,
        pydantic_model: openai.types.audio.SpeechCreateParams,
        *args,
        **kwargs,
):
    validate_tts(pydantic_model)
    router = get_router()

    speech_output: HttpxBinaryResponseContent = await router.aspeech(
        **pydantic_model
    )
    byte_stream = await speech_output.aiter_bytes()
    return StreamingHttpResponse(
        streaming_content=byte_stream,
        content_type='text/event-stream',
    )
