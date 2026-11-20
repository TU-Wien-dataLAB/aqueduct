from typing import Annotated, Any, Literal, cast

from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import BaseModel, Field, TypeAdapter

from gateway.decorators import (
    catch_router_exceptions,
    check_model_availability,
    check_model_concurrency_limits,
    check_model_rate_limits,
    check_token_concurrency_limits,
    check_token_rate_limits,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from gateway.response_type import RawJsonResponse, get_token_usage
from gateway.views.utils import oai_client_from_body
from management.models import Request, Token

ScalarOrStructured = str | dict[str, Any] | list[Any]


class NoulCriteria(BaseModel):
    true: ScalarOrStructured | None = None
    false: ScalarOrStructured | None = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: ScalarOrStructured | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: ScalarOrStructured | None = None
    criteria: dict[str, ScalarOrStructured | None] = Field(min_length=1, max_length=255)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: ScalarOrStructured | None = None
    criteria: list[ScalarOrStructured] = Field(min_length=1, max_length=10)


class BoxQuestion(BaseModel):
    type: Literal["box"]
    instructions: ScalarOrStructured | None = None


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | BoxQuestion, Field(discriminator="type")
]


class Asset(BaseModel):
    type: Literal["image"]
    format: Literal["png", "jpg", "jpeg"] | None = None
    data_base64: str | None = None
    part: str | None = None


class SystemOneRequest(BaseModel):
    model: str
    state: ScalarOrStructured
    questions: dict[str, Question] = Field(min_length=1)
    assets: dict[str, Asset] | None = None


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(SystemOneRequest))
@check_token_concurrency_limits
@check_model_concurrency_limits
@check_token_rate_limits
@check_model_rate_limits
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
    model = pydantic_model["model"]

    client, model_relay = oai_client_from_body(model, request)

    body = pydantic_model.copy()
    body["model"] = model_relay
    body.pop("timeout", None)

    response = cast("dict[str, Any]", await client.post("/systemone", cast_to=object, body=body))

    request_log.token_usage = get_token_usage(response)
    return RawJsonResponse(data=response, status=200)
