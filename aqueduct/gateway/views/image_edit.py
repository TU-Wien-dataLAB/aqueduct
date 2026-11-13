from typing import Any

from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm import BadRequestError
from openai.types import ImageEditParams
from pydantic import ConfigDict, RootModel, TypeAdapter

from gateway.config import get_model_supports_response_format
from gateway.decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from gateway.response_type import RawJsonResponse, get_token_usage
from gateway.views.utils import oai_client_from_body
from management.models import Request


class ImageEditBody(RootModel):  # type: ignore[type-arg]
    root: ImageEditParams
    model_config = ConfigDict(arbitrary_types_allowed=True)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(ImageEditBody), image_edit=True)
@check_limits
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def image_edit(
    request: ASGIRequest,
    pydantic_model: ImageEditParams,
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    model_name = pydantic_model.get("model") or ""
    if pydantic_model.get("stream"):
        raise BadRequestError(
            "Aqueduct does not support image streaming.", model_name, llm_provider=None
        )
    if pydantic_model.get("response_format", "b64_json") != "b64_json":
        raise BadRequestError(
            "Only b64_json response format is supported for image editing.",
            model_name,
            llm_provider=None,
        )
    if get_model_supports_response_format(model_name):
        pydantic_model["response_format"] = "b64_json"
    else:
        pydantic_model.pop("response_format", None)
    client, model_relay = oai_client_from_body(model_name, request)
    pydantic_model["model"] = model_relay
    try:
        response = await client.images.edit(**pydantic_model)
    except TypeError as err:
        raise BadRequestError(
            "Unexpected argument in request body", model_name, llm_provider=None
        ) from err
    request_log.token_usage = get_token_usage(response)
    return RawJsonResponse(response)
