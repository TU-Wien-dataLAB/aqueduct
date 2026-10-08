"""Shared type aliases for the gateway decorators package."""

from collections.abc import Callable, Coroutine
from typing import Any

from django.http import HttpResponse, StreamingHttpResponse

from gateway.response_type import RawJsonResponse, RawStreamingResponse

ViewResult = HttpResponse | StreamingHttpResponse | RawJsonResponse | RawStreamingResponse
AsyncView = Callable[..., Coroutine[Any, Any, ViewResult]]
Decorator = Callable[[AsyncView], AsyncView]

__all__ = ["AsyncView", "Decorator", "ViewResult"]
