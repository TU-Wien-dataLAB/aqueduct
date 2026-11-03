from typing import TYPE_CHECKING, Any

from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from litellm.types.rerank import RerankRequest
from pydantic import TypeAdapter

from gateway.config import get_router
from gateway.decorators import (
    catch_router_exceptions,
    check_limits,
    check_model_availability,
    ensure_usage,
    log_request,
    parse_body,
    resolve_alias,
    token_authenticated,
    tos_accepted,
)
from gateway.response_type import RawJsonResponse, get_token_usage
from management.models import Request

if TYPE_CHECKING:
    from litellm.types.rerank import RerankResponse


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(RerankRequest))
@check_limits
@ensure_usage
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def rerank(
    request: ASGIRequest,
    pydantic_model: RerankRequest,
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    router = get_router()
    response: RerankResponse = await router.arerank(**pydantic_model)
    request_log.token_usage = get_token_usage(response)
    return RawJsonResponse(data=response, status=200)
