from typing import TYPE_CHECKING, Any

import openai
from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from openai.types.decision import Decision

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
@parse_body(model=TypeAdapter(openai.types.decision_create_params.DecisionCreateParams))
@check_limits
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def decisions(
    request: ASGIRequest,
    pydantic_model: dict[str, Any],
    request_log: Request,
    token: Token,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    """Handler for POST /v1/decisions - Evaluates classification and scoring questions.

    This endpoint proxies the OpenAI Decisions API to the configured model
    (typically a decision model served via vLLM), forwarding the shared
    ``input`` and ``questions`` and returning the scored ``answers``.
    """

    model = pydantic_model.get("model")
    if not isinstance(model, str):
        return error_response("Missing required parameter: model", param="model", status=400)

    client, model_relay = oai_client_from_body(model, request)
    pydantic_model["model"] = model_relay

    decision: Decision = await client.decisions.create(**pydantic_model)

    request_log.token_usage = get_token_usage(decision)
    return RawJsonResponse(data=decision, status=200)
