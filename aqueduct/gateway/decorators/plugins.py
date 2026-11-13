import logging
from functools import wraps
from typing import Any

from django.core.handlers.asgi import ASGIRequest

from gateway.decorators.types import AsyncView, ViewResult
from gateway.response_error import error_response
from gateway.response_type import RawStreamingResponse
from management.plugins import (
    BlockedByPlugin,
    PluginSnippet,
    _error_hook,
    after_hook,
    before_hook,
    stream_hook,
)

log = logging.getLogger("aqueduct")


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
            response.transforms.append(lambda chunk: stream_hook(plugins, chunk))

        after_hook(plugins, request, token, response)
        return response

    return wrapper


def plugin_errors(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        plugins: list[PluginSnippet] = getattr(request, "active_plugins", [])
        if not plugins:
            return await view_func(request, *args, **kwargs)
        try:
            return await view_func(request, *args, **kwargs)
        except Exception as exc:
            try:
                _error_hook(plugins, request, exc)
            except Exception:
                log.exception("Plugin on_error hook raised while handling %r", exc)
            raise

    return wrapper
