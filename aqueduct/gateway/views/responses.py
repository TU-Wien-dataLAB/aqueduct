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
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
    validate_response_id,
)
from .utils import (
    ResponseRegistrationWrapper,
    _get_token_usage,
    _openai_stream,
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
@log_request
@resolve_alias
@check_model_availability
@catch_router_exceptions
async def create_response(
    request: ASGIRequest, pydantic_model: dict, request_log: Request, token: Token, *args, **kwargs
):
    """Handler for POST /v1/responses - Creates a new response via OpenAI's responses API

    This endpoint forwards requests to the OpenAI responses API, handling both streaming
    and non-streaming responses."""

    model = pydantic_model.get("model")
    client, model_relay = oai_client_from_body(pydantic_model.get("model"), request)
    pydantic_model["model"] = model_relay

    resp: Response | AsyncStream = await client.responses.create(**pydantic_model)

    if isinstance(resp, AsyncStream):
        return StreamingHttpResponse(
            streaming_content=ResponseRegistrationWrapper(
                streaming_content=_openai_stream(stream=resp, request_log=request_log),
                model=model,
                email=token.user.email,
            ),
            content_type="text/event-stream",
        )
    elif isinstance(resp, Response):
        register_response_in_cache(resp.id, model=model, email=token.user.email)
        data = resp.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _get_token_usage(data)
        return JsonResponse(data=data, status=200)
    else:
        raise NotImplementedError(f"Completion for response type {type(resp)} is not implemented.")


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
@validate_response_id
@catch_router_exceptions
async def response(request: ASGIRequest, response_id: str, token: Token, *args, **kwargs):
    """Combined handler for GET and DELETE /v1/responses/{response_id}"""
    response = get_response_from_cache(response_id)
    model = response["model"]
    client, model_relay = oai_client_from_body(model, request)

    if request.method == "GET":
        return await client.responses.retrieve(response_id=response_id)
    elif request.method == "DELETE":
        return await client.responses.delete(response_id=response_id)


@csrf_exempt
@require_GET
@token_authenticated(token_auth_only=True)
@tos_accepted
@log_request
@validate_response_id
@catch_router_exceptions
async def get_response_input_items(
    request: ASGIRequest, response_id: str, token: Token, *args, **kwargs
):
    """Handler for GET /v1/responses/{response_id}/input_items"""
    response = get_response_from_cache(response_id)
    model = response["model"]
    client, model_relay = oai_client_from_body(model, request)
    return await client.responses.input_items.list(response_id=response_id)
