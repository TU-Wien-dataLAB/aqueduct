import litellm
import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter, ConfigDict, RootModel

from gateway.router import get_openai_client, get_router
from management.models import Request
from .decorators import (
    token_authenticated,
    check_limits,
    parse_body,
    log_request,
    check_model_availability,
    catch_router_exceptions,
)
from .utils import _get_token_usage, _openai_stream


class TranscriptionCreateParams(RootModel):
    root: openai.types.audio.TranscriptionCreateParams
    # IO[bytes] requires arbitrary_types_allowed for model settings
    model_config = ConfigDict(arbitrary_types_allowed=True)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@check_limits
@parse_body(model=TypeAdapter(TranscriptionCreateParams))
@log_request
@check_model_availability
@catch_router_exceptions
async def transcriptions(
        request: ASGIRequest,
        pydantic_model: openai.types.audio.TranscriptionCreateParams,
        request_log: Request,
        *args,
        **kwargs,
):
    model: str = pydantic_model.get('model', 'unknown')
    try:
        client: openai.AsyncClient = get_openai_client(model)
    except ValueError:
        return JsonResponse({"error": f"Incompatible model '{model}'!"}, status=400)

    router = get_router()
    deployment: litellm.Deployment = router.get_deployment(model_id=model)

    model_relay, provider, _, _ = litellm.get_llm_provider(deployment.litellm_params.model)
    pydantic_model['model'] = model_relay

    transcription = await client.audio.transcriptions.create(**pydantic_model)


    if (isinstance(transcription, openai.types.audio.transcription.Transcription) or
            isinstance(transcription, openai.types.audio.transcription_verbose.TranscriptionVerbose)):
        data = transcription.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _get_token_usage(data)
        return JsonResponse(data=data, status=200)
    elif isinstance(transcription, openai.AsyncStream):
        return StreamingHttpResponse(
            streaming_content=_openai_stream(stream=transcription, request_log=request_log),
            content_type='text/event-stream',
        )
    else:
        raise RuntimeError(f"Received unexpected response type: {type(transcription)}")
