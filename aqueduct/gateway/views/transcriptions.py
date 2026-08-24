from typing import Any

import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from openai.types.audio.transcription_create_params import (
    TranscriptionCreateParams as OpenAITranscriptionCreateParams,
)
from pydantic import ConfigDict, RootModel, TypeAdapter

from management.models import Request

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from .utils import RawJsonResponse, RawStreamingResponse, get_token_usage, oai_client_from_body


class TranscriptionCreateParams(RootModel):  # type: ignore[type-arg]
    root: OpenAITranscriptionCreateParams
    # IO[bytes] requires arbitrary_types_allowed for model settings
    model_config = ConfigDict(arbitrary_types_allowed=True)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(TranscriptionCreateParams))
@check_limits
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def transcriptions(
    request: ASGIRequest,
    pydantic_model: OpenAITranscriptionCreateParams,
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse | HttpResponse | RawStreamingResponse:
    client, model_relay = oai_client_from_body(pydantic_model.get("model"), request)
    pydantic_model["model"] = model_relay

    transcription = await client.audio.transcriptions.create(**pydantic_model)

    if isinstance(
        transcription,
        (
            openai.types.audio.transcription.Transcription,
            openai.types.audio.transcription_verbose.TranscriptionVerbose,
        ),
    ):
        data = transcription.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = get_token_usage(data)
        return RawJsonResponse(data=data, status=200)
    if isinstance(transcription, str):
        # Text-based formats (VTT, SRT, text) return plain strings
        return HttpResponse(
            content=transcription.encode("utf-8"),
            content_type="text/plain; charset=utf-8",
            status=200,
        )
    if isinstance(transcription, openai.AsyncStream):
        return RawStreamingResponse(streaming_content=transcription, request_log=request_log)
    raise RuntimeError(f"Received unexpected response type: {type(transcription)}")
