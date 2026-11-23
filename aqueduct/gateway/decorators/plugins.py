from functools import wraps
from typing import Any, cast

from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, ViewResult
from gateway.response_error import error_response
from gateway.response_type import RawStreamingResponse
from management.plugins import (
    BlockedByPlugin,
    PluginSnippet,
    after_hook,
    after_streaming_hook,
    before_hook,
)


def run_plugins(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        plugins: list[PluginSnippet] = getattr(request, "active_plugins", [])
        if not plugins:
            return await view_func(request, *args, **kwargs)

        token = kwargs.get("token")
        body = kwargs.get("pydantic_model")

        try:
            final_body = before_hook(plugins, request, token, body)
        except BlockedByPlugin as e:
            return error_response(e.reason, status=e.status)

        if final_body is not None:
            kwargs["pydantic_model"] = final_body

        response = await view_func(request, *args, **kwargs)

        if isinstance(response, RawStreamingResponse):
            return cast("ViewResult", after_streaming_hook(plugins, request, response))
        return cast("ViewResult", after_hook(plugins, request, response))

    return wrapper
