import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, ViewResult
from gateway.raw_response import error_response

if TYPE_CHECKING:
    from management.models import Token

log = logging.getLogger("aqueduct")


def check_model_availability(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_model_availability decorator used without @token_authenticated decorator"
            )
            return error_response("Internal server error", status=500)
        body: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not body:
            return await view_func(request, *args, **kwargs)
        model: str | None = body.get("model", None)
        if not model:
            return await view_func(request, *args, **kwargs)
        if await sync_to_async(token.model_excluded)(model):
            log.error("Model not found - %s", model)
            return error_response("Model not found!", status=404)
        return await view_func(request, *args, **kwargs)

    return wrapper
