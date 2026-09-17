from typing import TYPE_CHECKING, Any

import openai
from django.core.handlers.asgi import ASGIRequest
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from pydantic import TypeAdapter

from gateway.config import get_router
from gateway.decorators.auth import token_authenticated, tos_accepted
from gateway.decorators.availability import check_model_availability
from gateway.decorators.body import ensure_usage, parse_body, resolve_alias
from gateway.decorators.errors import catch_router_exceptions
from gateway.decorators.limits import check_limits
from gateway.decorators.log import log_request
from gateway.response_types import RawJsonResponse
from gateway.views.utils import get_token_usage
from management.models import Request

if TYPE_CHECKING:
    from litellm.types.utils import EmbeddingResponse


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
@tos_accepted
@parse_body(model=TypeAdapter(openai.types.EmbeddingCreateParams))
@check_limits
@ensure_usage
@resolve_alias
@check_model_availability
@log_request
@catch_router_exceptions
async def embeddings(
    request: ASGIRequest,
    pydantic_model: openai.types.EmbeddingCreateParams,
    request_log: Request,
    *args: Any,
    **kwargs: Any,
) -> RawJsonResponse:
    router = get_router()
    embedding: EmbeddingResponse = await router.aembedding(**pydantic_model)
    request_log.token_usage = get_token_usage(embedding)
    return RawJsonResponse(data=embedding, status=200)
