import logging
from functools import wraps
from typing import TYPE_CHECKING, Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest

from gateway.config import get_public_model_name
from gateway.decorators.types import AsyncView, ViewResult
from gateway.model_rate_limiting import check_model_limit
from gateway.rate_limiting import check_and_reserve, has_any_limit
from gateway.response_error import error_response

if TYPE_CHECKING:
    from management.models import Token

log = logging.getLogger("aqueduct")


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
                    messages = (
                        f"{message} (Retry after {seconds_left}s)"
                        for message, seconds_left in exceeded
                    )
                    error_message = "Rate limit exceeded. " + ", ".join(messages) + "."
                    log.warning(
                        "Rate limit exceeded for Token %r (ID: %s). Details: %s",
                        token.name,
                        token.id,
                        error_message,
                    )
                    log.error("Rate limit exceeded - %s", error_message)
                    # Wait until every exceeded bucket has ended
                    retry_after = max(seconds_left for _message, seconds_left in exceeded)
                    # Return 429 Too Many Requests
                    return error_response(
                        error_message, status=429, headers={"Retry-After": str(retry_after)}
                    )

            model = (kwargs.get("pydantic_model") or {}).get("model")
            if model and settings.AQUEDUCT_MODEL_RPM_LIMITS:
                model = get_public_model_name(model)
                if model is None:
                    return error_response(
                        "Unknown public model. Use a configured model name or its Aqueduct alias.",
                        param="model",
                        status=400,
                    )
                rpm = settings.AQUEDUCT_MODEL_RPM_LIMITS.get(model)
                if rpm is not None:
                    # Do not let excluded tokens consume capacity shared by all users.
                    if await sync_to_async(token.model_excluded)(model):
                        return error_response("Model not found!", status=404)
                    allowed, retry_after = await sync_to_async(check_model_limit)(model, rpm)
                    if not allowed:
                        return error_response(
                            f"Model request limit ({rpm}/min) exceeded for '{model}'.",
                            status=429,
                            headers={"Retry-After": str(retry_after)},
                        )

        except Exception as e:
            log.exception("Error checking rate limits for Token %r: %s", token.name, e)
            return error_response("Internal gateway error checking rate limits", status=500)

        return await view_func(request, *args, **kwargs)

    return wrapper
