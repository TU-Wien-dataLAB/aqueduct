import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from functools import wraps
from typing import Any, cast

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http import StreamingHttpResponse
from django.utils import timezone

from gateway.decorators.types import AsyncView, ViewResult
from gateway.rate_limiting import record_token_usage
from gateway.response_type import release_db_connection
from management.models import Request

log = logging.getLogger("aqueduct")


async def _guard_stream(stream: AsyncIterator[bytes], path: str) -> AsyncGenerator[bytes]:
    """Stream without holding a DB connection, and end quietly on errors (#246).

    An exception escaping a stream skips Django's `request_finished`, so the
    request's DB connection is never returned to the pool. Cancellation and
    `GeneratorExit` still propagate.
    """
    await sync_to_async(release_db_connection)()
    try:
        async for chunk in stream:
            yield chunk
    except Exception:
        log.exception("Error while streaming response for %s", path)


def client_ip(meta: dict[str, Any]) -> str | None:
    """The client IP from ``CLIENT_IP_HEADERS`` (first non-empty wins),
    falling back to ``REMOTE_ADDR``."""
    for key in settings.CLIENT_IP_HEADERS:
        value: str | None = meta.get(key)
        if value:
            return value
    return meta.get("REMOTE_ADDR")


def log_request(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        token = kwargs.get("token")
        request_log = Request(
            token=token,
            model="" if not pydantic_model else pydantic_model.get("model", ""),
            # Use the resolved model from self
            timestamp=timezone.now(),
            method=request.method,
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=client_ip(request.META),
            user_id=kwargs.get("user_id", ""),
            # path, Status, time, usage set later in the view or processing steps
        )
        # Calculate and set path (ensure leading slash)
        request_log.path = f"/{request.path.lstrip('/')}"
        kwargs["request_log"] = request_log
        await request_log.asave()
        log.debug("Initial request log object created.")

        response_start_time = time.monotonic()
        try:
            result: ViewResult = await view_func(request, *args, **kwargs)
        except asyncio.CancelledError:
            request_log.processing_time_ms = int(
                (response_start_time - kwargs["request_start"]) * 1000
            )
            request_log.response_time_ms = int((time.monotonic() - response_start_time) * 1000)
            request_log.status_code = 499
            await request_log.asave()
            raise

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
        else:
            streaming_content = cast("AsyncIterator[bytes]", result.streaming_content)
            result.streaming_content = _guard_stream(streaming_content, request_log.path)

        return result

    return wrapper
