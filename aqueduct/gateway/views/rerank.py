import logging
from typing import Any, NotRequired, TypedDict

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
from management.models import Request

log = logging.getLogger("aqueduct")


class RerankRequestParams(TypedDict):
    """Jina-style rerank request body, proxied to the upstream endpoint.

    Served by OpenAI-compatible backends (llama.cpp / vLLM) under ``/v1/rerank``.
    """

    model: str
    query: str
    documents: list[str | dict[str, Any]]
    top_n: NotRequired[int]
    return_documents: NotRequired[bool]
    rank_fields: NotRequired[list[str]]


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(RerankRequestParams))
@check_limits
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def rerank(
    request: ASGIRequest,
    pydantic_model: dict[str, Any],
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    """Handler for POST /v1/rerank - Ranks documents by relevance to a query.

    Proxies the Jina-style rerank request to the configured model's
    OpenAI-compatible backend (``{api_base}/rerank``), forwarding ``query``
    and ``documents`` and returning the scored ``results``.
    """
    model = pydantic_model["model"]

    client, model_relay = oai_client_from_body(model, request)
    pydantic_model["model"] = model_relay

    # Drop the injected timeout and unset optional fields; the upstream
    # rerank API does not accept them.
    body = {k: v for k, v in pydantic_model.items() if v is not None and k != "timeout"}
    # Raw passthrough: no typed rerank resource in the OpenAI SDK.
    response: Any = await client.post("rerank", cast_to=object, body=body)
    if not isinstance(response, dict):
        log.error("Unexpected rerank response from upstream: %r", response)
        return error_response("Malformed rerank response from the model provider.", status=502)

    request_log.token_usage = get_token_usage(response)
    return RawJsonResponse(data=response, status=200)
