from typing import Any, Literal, cast

from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import BaseModel, ConfigDict, TypeAdapter

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


class ChoiceOption(BaseModel):
    """A selectable value for a ``choice`` question."""

    model_config = ConfigDict(extra="allow")
    value: str
    description: str = ""


class ScoreLevel(BaseModel):
    """An ordered level for a ``score`` question."""

    model_config = ConfigDict(extra="allow")
    label: str
    description: str = ""


class DecisionQuestion(BaseModel):
    """A single question in a Decisions API request."""

    model_config = ConfigDict(extra="allow")
    type: Literal["predicate", "choice", "score"]
    name: str
    instructions: str = ""
    choices: list[ChoiceOption] | None = None
    levels: list[ScoreLevel] | None = None


class DecisionCreateParams(BaseModel):
    """Request body for POST /v1/decisions."""

    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[Any]
    questions: list[DecisionQuestion]


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(DecisionCreateParams))
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
    """Handler for POST /v1/decisions (and /v1/decision)"""

    model = pydantic_model.get("model")
    if not isinstance(model, str):
        return error_response("Missing required parameter: model", param="model", status=400)

    client, _model_relay = oai_client_from_body(model, request)
    body = {k: v for k, v in pydantic_model.items() if k not in ("timeout", "user_id")}

    result = await client.post("/decisions", cast_to=object, body=body)
    result_dict = cast("dict[str, Any]", result)

    request_log.token_usage = get_token_usage(result_dict)
    return RawJsonResponse(data=result_dict, status=200)
