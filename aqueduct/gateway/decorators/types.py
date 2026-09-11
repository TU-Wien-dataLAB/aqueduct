from collections.abc import Callable, Coroutine
from typing import Any

from django.http import HttpResponse, StreamingHttpResponse

from gateway.raw_response import RawJsonResponse, RawStreamingResponse

ViewResult = HttpResponse | StreamingHttpResponse | RawJsonResponse | RawStreamingResponse
AsyncView = Callable[..., Coroutine[Any, Any, ViewResult]]
Decorator = Callable[[AsyncView], AsyncView]
