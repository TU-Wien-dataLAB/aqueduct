"""Rate-limit checking decorator."""

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, ViewResult
from gateway.rate_limiting import check_and_reserve, has_any_limit
from gateway.response_error import error_response

if TYPE_CHECKING:
    from management.models import Token

log = logging.getLogger("aqueduct")

__all__ = ["check_limits"]


def check_limits(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_limits decorator used without @token_authenticated decorator")
            return error_response("Internal server error", status=500)

        try:
            # Get limits asynchronously
            limits = await sync_to_async(token.get_limit)()
            log.debug("Rate limits for Token %r (ID: %s): %s", token.name, token.id, limits)

            if settings.AQUEDUCT_RATE_LIMIT_ENABLED and has_any_limit(limits):
                model = (kwargs.get("pydantic_model") or {}).get("model")
                allowed, exceeded = await sync_to_async(check_and_reserve)(limits, token.id, model)
                if not allowed:
                    error_message = "Rate limit exceeded. " + ", ".join(exceeded) + "."
                    log.warning(
                        "Rate limit exceeded for Token %r (ID: %s). Details: %s",
                        token.name,
                        token.id,
                        error_message,
                    )
                    log.error("Rate limit exceeded - %s", error_message)
                    # Return 429 Too Many Requests
                    return error_response(error_message, status=429)

        except Exception as e:
            log.exception("Error checking rate limits for Token %r: %s", token.name, e)
            return error_response("Internal gateway error checking rate limits", status=500)

        return await view_func(request, *args, **kwargs)

    return wrapper
