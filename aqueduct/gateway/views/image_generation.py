import logging

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm import BadRequestError
from openai.types import ImageGenerateParams, ImagesResponse
from pydantic import ConfigDict, TypeAdapter

from management.models import Request

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
)
from .utils import _get_token_usage, oai_client_from_body

log = logging.getLogger("aqueduct")


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@check_limits
@parse_body(model=TypeAdapter(ImageGenerateParams, config=ConfigDict(extra="forbid")))
@log_request
@resolve_alias
@check_model_availability
@catch_router_exceptions
async def image_generation(
    request: ASGIRequest, pydantic_model: ImageGenerateParams, request_log: Request, *args, **kwargs
):
    if pydantic_model.get("stream"):
        # LiteLLM cannot parse a Stream response, so we don't support streaming for now
        raise BadRequestError(
            "Aqueduct does not support image streaming.",
            pydantic_model.get("model"),
            llm_provider=None,
        )

    response_format = pydantic_model.get("response_format", "b64_json")
    if response_format != "b64_json":
        raise BadRequestError(
            "Only b64_json response format is supported for image generation.",
            pydantic_model.get("model"),
            llm_provider=None,
        )
    # update if it was None
    pydantic_model["response_format"] = response_format

    client, model_relay = oai_client_from_body(pydantic_model.get("model"), request)
    pydantic_model["model"] = model_relay

    try:
        resp: ImagesResponse = await client.images.generate(**pydantic_model)
    except TypeError as err:
        # Sending extra fields in the data makes `AsyncImages.generate()` error out
        log.error(err)
        raise BadRequestError(
            "Unexpected argument in request body", pydantic_model.get("model"), llm_provider=None
        )

    data = resp.model_dump(exclude_unset=True)
    request_log.token_usage = _get_token_usage(data)

    return JsonResponse(data)
