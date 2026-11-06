from typing import Any, cast

from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter

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
from gateway.response_error import error_response
from gateway.response_type import RawJsonResponse, get_token_usage
from gateway.views.utils import oai_client_from_body
from management.models import Request, Token


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(dict))
@check_limits
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def systemone(
    request: ASGIRequest,
    pydantic_model: dict[str, Any],
    request_log: Request,
    token: Token,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    model = pydantic_model.get("model")
    if not isinstance(model, str):
        return error_response("Missing required parameter: model", param="model", status=400)

    client, model_relay = oai_client_from_body(model, request)
    pydantic_model["model"] = model_relay

    body = dict(pydantic_model)
    body.pop("timeout", None)

    response = cast("dict[str, Any]", await client.post("/systemone", cast_to=object, body=body))

    request_log.token_usage = get_token_usage(response)
    return RawJsonResponse(data=response, status=200)
