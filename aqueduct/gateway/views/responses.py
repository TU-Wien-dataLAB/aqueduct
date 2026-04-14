from typing import Any

import openai
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from openai import AsyncStream
from openai.types.responses import Response
from pydantic import TypeAdapter

from management.models import Request, Token

from .decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    check_tool_availability,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
    validate_response_id,
)
from .errors import error_response
from .utils import (
    ResponseRegistrationWrapper,
    _get_token_usage,
    _openai_stream,
    delete_response_from_cache,
    get_response_from_cache,
    oai_client_from_body,
    register_response_in_cache,
)


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@check_limits
@parse_body(model=TypeAdapter(openai.types.responses.ResponseCreateParams))
@tos_accepted
@resolve_alias
@log_request
@check_model_availability
@check_tool_availability
@catch_router_exceptions
async def create_response(
    request: ASGIRequest,
    pydantic_model: dict[str, Any],
    request_log: Request,
    token: Token,
    *args: Any,
    **kwargs: Any,
) -> JsonResponse | StreamingHttpResponse:
    """Handler for POST /v1/responses - Creates a new response via OpenAI's responses API

    This endpoint forwards requests to the OpenAI responses API, handling both streaming
    and non-streaming responses."""

    model = pydantic_model.get("model")
    if not isinstance(model, str):
        return error_response("Missing required parameter: model", param="model", status=400)
    client, model_relay = oai_client_from_body(model, request)
    pydantic_model["model"] = model_relay

    resp: Response | AsyncStream[Response] = await client.responses.create(**pydantic_model)

    if isinstance(resp, AsyncStream):
        return StreamingHttpResponse(
            streaming_content=ResponseRegistrationWrapper(
                streaming_content=_openai_stream(stream=resp, request_log=request_log),
                model=model,
                email=token.user.email,
            ),
            content_type="text/event-stream",
        )
    if isinstance(resp, Response):
        register_response_in_cache(resp.id, model=model, email=token.user.email)
        data = resp.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _get_token_usage(data)
        return JsonResponse(data=data, status=200)
    raise NotImplementedError(f"Completion for response type {type(resp)} is not implemented.")


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
@validate_response_id
@catch_router_exceptions
async def response(
    request: ASGIRequest, response_id: str, token: Token, *args: Any, **kwargs: Any
) -> JsonResponse:
    """Combined handler for GET and DELETE /v1/responses/{response_id}"""
    response = get_response_from_cache(response_id)
    if response is None:
        return JsonResponse({"error": "Response not found"}, status=404)
    model: str = response["model"]
    client, _model_relay = oai_client_from_body(model, request)

    if request.method == "GET":
        get_resp = await client.responses.retrieve(response_id=response_id)
        data = get_resp.model_dump(exclude_none=True, exclude_unset=True)
        return JsonResponse(data=data, status=200)
    if request.method == "DELETE":
        delete_result = await client.responses.delete(response_id=response_id)  # type: ignore[func-returns-value]
        delete_response_from_cache(response_id=response_id)
        if delete_result is None:
            # BUG in openai python sdk: https://github.com/openai/openai-openapi/issues/490
            return JsonResponse({"deleted": True}, status=200)
        data = delete_result.model_dump(exclude_none=True, exclude_unset=True)
        return JsonResponse(data=data, status=200)
    raise AssertionError("Unreachable")


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
@validate_response_id
@catch_router_exceptions
async def get_response_input_items(
    request: ASGIRequest, response_id: str, token: Token, *args: Any, **kwargs: Any
) -> JsonResponse:
    """Handler for GET /v1/responses/{response_id}/input_items"""
    response = get_response_from_cache(response_id)
    if response is None:
        return JsonResponse({"error": "Response not found"}, status=404)
    model: str = response["model"]
    client, _model_relay = oai_client_from_body(model, request)
    resp = await client.responses.input_items.list(response_id=response_id)
    data = resp.model_dump(exclude_none=True, exclude_unset=True)
    return JsonResponse(data=data, status=200)
