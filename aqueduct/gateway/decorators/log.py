"""Request logging decorator."""

import logging
import time
from functools import wraps
from typing import Any

from asgiref.sync import sync_to_async
from django.core.handlers.asgi import ASGIRequest
from django.http import StreamingHttpResponse
from django.utils import timezone

from gateway.decorators.types import AsyncView, ViewResult
from gateway.rate_limiting import record_token_usage
from management.models import Request

log = logging.getLogger("aqueduct")

__all__ = ["log_request"]


def log_request(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        is_initialize = kwargs.get("is_initialize", False)

        if request.path.startswith("/mcp-servers/") and not is_initialize:
            kwargs["request_log"] = None
            return await view_func(request, *args, **kwargs)

        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        token = kwargs.get("token")
        request_log = Request(
            token=token,
            model="" if not pydantic_model else pydantic_model.get("model", ""),
            # Use the resolved model from self
            timestamp=timezone.now(),
            method=request.method,
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.META.get("REMOTE_ADDR"),
            user_id=kwargs.get("user_id", ""),
            # path, Status, time, usage set later in the view or processing steps
        )
        # Calculate and set path (ensure leading slash)
        request_log.path = f"/{request.path.lstrip('/')}"
        kwargs["request_log"] = request_log
        await request_log.asave()
        log.debug("Initial request log object created.")

        response_start_time = time.monotonic()
        result: ViewResult = await view_func(request, *args, **kwargs)
        end_time = time.monotonic()

        assert "request_start" in kwargs, (
            "`log_request` decorator can only be used with the `token_authenticated` decorator"
        )
        request_log.processing_time_ms = int((response_start_time - kwargs["request_start"]) * 1000)
        request_log.response_time_ms = int((end_time - response_start_time) * 1000)
        request_log.status_code = result.status_code

        await request_log.asave()

        # Record token usage into the rate-limit buckets for non-streaming
        # responses. For non-streaming responses the view has finalized
        # ``request_log.token_usage`` before returning, so it is available here.
        # Streaming responses defer token recording to
        # ``RawStreamingResponse._iter_stream`` (the generator runs after this
        # wrapper returns), so we skip them here to avoid double-counting.
        if not isinstance(result, StreamingHttpResponse):
            await sync_to_async(record_token_usage)(request_log.token_id, request_log.token_usage)

        return result

    return wrapper
