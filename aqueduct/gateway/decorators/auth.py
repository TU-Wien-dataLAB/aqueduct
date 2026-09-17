import logging
import time
from functools import wraps
from typing import Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.cache import cache
from django.core.handlers.asgi import ASGIRequest
from tos.models import has_user_agreed_latest_tos  # type: ignore[import-untyped]

from gateway.authentication import token_from_request
from gateway.decorators.types import AsyncView, Decorator, ViewResult
from gateway.response_types import error_response
from management.models import Token

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
