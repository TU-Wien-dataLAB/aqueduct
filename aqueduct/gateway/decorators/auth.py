import logging
import time
from datetime import timedelta
from functools import wraps
from typing import Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.cache import cache
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Sum
from django.utils import timezone
from tos.models import has_user_agreed_latest_tos  # type: ignore[import-untyped]

from gateway.authentication import token_from_request
from gateway.config import get_all_model_request_limit_multipliers, resolve_model_alias
from gateway.decorators.types import AsyncView, Decorator, ViewResult
from gateway.raw_response import error_response
from management.models import Request, Token

log = logging.getLogger("aqueduct")


def token_authenticated(token_auth_only: bool) -> Decorator:
    """
    Retrieve user's token from the database and add it to the decorated view's kwargs.

    Also register the time when the request came in, as "request_start" in kwargs;
    this value can later be used to measure the total time it took to process the request.

    Args:
        token_auth_only: if `True`, decorated view is only accessible with token
            authentication, i.e. token has to be sent in the request header, otherwise
            authentication fails.
            If `False`, authentication with other backends is also accepted.
    """

    def decorator(view_func: AsyncView) -> AsyncView:
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
            kwargs["request_start"] = time.monotonic()

            unauthorized_response = error_response("Authentication Required", status=401)
            # Authentication Check
            if not (await request.auser()).is_authenticated:
                user = await auth.aauthenticate(request=request)
                if user is not None:
                    request.user = user  # Manually assign user
            else:
                request.user = await request.auser()

            if not getattr(request, "user", None) or not request.user.is_authenticated:
                log.error(
                    "Authentication check failed in ai_gateway_view: "
                    "request.user is not authenticated."
                )
                return unauthorized_response
            log.debug("User %s authenticated.", request.user.email)

            token_key = token_from_request(request)
            if token_auth_only and not token_key:
                log.error("Token not found in request")
                return unauthorized_response

            if token_key:
                token = await sync_to_async(Token.find_by_key)(token_key)
            else:
                # user authenticated but not via token -> use Token.objects for async ORM
                token = (
                    await Token.objects.select_related(
                        "user__profile__org", "service_account__team__org"
                    )
                    .filter(user=request.user)
                    .afirst()
                )

            if not token:
                log.error("Token not found during authentication")
                return unauthorized_response
            kwargs["token"] = token
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


def tos_accepted(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        if settings.TOS_ENABLED and settings.TOS_GATEWAY_VALIDATION:
            token: Token | None = kwargs.get("token")
            if not token:
                log.error("tos_accepted decorator used without @token_authenticated decorator")
                return error_response("Internal server error", status=500)
            key_version = cache.get("django:tos:key_version")
            user_id = token.user.id

            skip: bool = cache.get(
                f"django:tos:skip_tos_check:{user_id}", False, version=key_version
            )

            if not skip:
                user_agreed = cache.get(f"django:tos:agreed:{user_id}", None, version=key_version)
                if user_agreed is None:
                    user_agreed = await sync_to_async(has_user_agreed_latest_tos)(request.user)

                if not user_agreed:
                    log.error("Terms of service agreement required")
                    return error_response(
                        "In order to use the API you have to agree to the terms of service!",
                        status=403,
                    )

        return await view_func(request, *args, **kwargs)

    return wrapper


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

            if (
                limits.requests_per_minute is not None
                or limits.input_tokens_per_minute is not None
                or limits.output_tokens_per_minute is not None
            ):
                # Define the time window for usage check (last 60 seconds)
                time_window_start = timezone.now() - timedelta(seconds=60)

                # Build queryset for recent requests (last 60 seconds)
                recent_requests = Request.objects.filter(
                    token=token, timestamp__gte=time_window_start
                )

                # Query recent usage asynchronously using Django's async ORM
                # Get overall token counts
                recent_requests_agg = await recent_requests.aaggregate(
                    request_count=Count("id"),
                    total_input_tokens=Sum("input_tokens"),
                    total_output_tokens=Sum("output_tokens"),
                )

                total_input = recent_requests_agg.get("total_input_tokens", 0) or 0
                total_output = recent_requests_agg.get("total_output_tokens", 0) or 0

                # Get per-model request counts for weighted budget calculation
                model_counts = {
                    item["model"]: item["request_count"]
                    async for item in (
                        recent_requests.exclude(model="")
                        .values("model")
                        .annotate(request_count=Count("id"))
                    )
                }

                log.debug(
                    "Recent usage (last 60s) for Token %r: Model counts=%s, Input=%s, Output=%s",
                    token.name,
                    model_counts,
                    total_input,
                    total_output,
                )

                # --- Check Limits ---
                exceeded = []

                # Calculate weighted request count using per-model multipliers
                # "2x Limits" means multiplier=2, so cost = 1/2 = 0.5 per request
                weighted_request_count: float = 0.0
                multipliers = get_all_model_request_limit_multipliers()
                for model, count in model_counts.items():
                    multiplier = multipliers.get(model, 1.0)
                    weighted_request_count += count * (1.0 / multiplier)

                log.debug(
                    "Weighted request count for Token %r: %.2f (base limit: %s)",
                    token.name,
                    weighted_request_count,
                    limits.requests_per_minute,
                )

                if (
                    limits.requests_per_minute is not None
                    and weighted_request_count >= limits.requests_per_minute
                ):
                    request_limit = float(limits.requests_per_minute)
                    pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
                    model = pydantic_model.get("model") if pydantic_model else None
                    if model:
                        request_limit *= multipliers.get(resolve_model_alias(model), 1.0)
                    exceeded.append(f"Request limit ({request_limit:g}/min)")

                if (
                    limits.input_tokens_per_minute is not None
                    and total_input >= limits.input_tokens_per_minute
                ):
                    exceeded.append(f"Input token limit ({limits.input_tokens_per_minute}/min)")

                if (
                    limits.output_tokens_per_minute is not None
                    and total_output >= limits.output_tokens_per_minute
                ):
                    exceeded.append(f"Output token limit ({limits.output_tokens_per_minute}/min)")

                if exceeded:
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
