import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, cast

import redis.asyncio as aioredis
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http.response import StreamingHttpResponse

from gateway.config import get_model_max_parallel_requests, resolve_model_alias
from gateway.decorators.types import AsyncView, ViewResult
from gateway.rate_limiting import check_and_reserve
from gateway.response_error import error_response

if TYPE_CHECKING:
    from management.models import Token


log = logging.getLogger("aqueduct")


# Lua: increment the counter for the `KEYS[1]` key, or undo-and-reject if over limit.
# ARGV[1] = limit, ARGV[2] = TTL: the key auto-resets after this time in case the backend
# failed to call the release script for whatever reason (i.e. worker failure).
ACQUIRE_LUA = """
local c = redis.call('INCR', KEYS[1])
if c > tonumber(ARGV[1]) then
    redis.call('DECR', KEYS[1])
    return 0
end
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""


# Lua: decrement the counter for the `KEYS[1]` key, making sure it does not become negative.
RELEASE_LUA = """
local c = redis.call('GET', KEYS[1])
if c and tonumber(c) > 0 then
    redis.call('DECR', KEYS[1])
    return 1
end
return 0
"""


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    async_redis = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return async_redis


@lru_cache(maxsize=1)
def get_limiter() -> "RedisConcurrencyLimiter":
    return RedisConcurrencyLimiter(get_redis_client())


def token_concurrency_key(token: "Token") -> str:
    return f"token:concurrency:{token.id}"


def token_concurrency_limit(token: "Token") -> int:
    """Returns concurrent requests limit for the token."""
    if token.concurrent_requests_limit is not None:
        # Short-circuit if the limit is set on the token to avoid db queries
        return token.concurrent_requests_limit
    return token.get_limits().concurrent_requests_limit


def model_concurrency_key(model_name: str) -> str:
    return f"model:concurrency:{resolve_model_alias(model_name)}"


def model_concurrency_limit(model: str) -> int:
    """Returns the concurrent requests limit for a specific model.

    Looks up an optional per-model override defined in
    ``AQUEDUCT_MODEL_CONCURRENT_REQUESTS_LIMITS``,
    falling back to the default model concurrency limit.
    """
    configured = get_model_max_parallel_requests(model)
    if configured is not None:
        return configured
    return settings.AQUEDUCT_DEFAULT_MODEL_CONCURRENT_REQUESTS_LIMIT


class RedisConcurrencyLimiter:
    def __init__(self, redis_client: aioredis.Redis, refresh_interval: int = 30):
        self.redis = redis_client
        self.acquire_script = redis_client.register_script(ACQUIRE_LUA)
        self.release_script = redis_client.register_script(RELEASE_LUA)
        self.ttl = settings.AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS
        # Refresh interval has to be safely smaller than ttl; otherwise slots of long-running
        # stream requests may expire before the streaming is over!
        self.refresh_interval = min(refresh_interval, self.ttl // 2)

    async def acquire(self, key: str, limit: int) -> bool:
        ok = await self.acquire_script(keys=[key], args=[limit, self.ttl])
        return bool(ok)

    async def release(self, key: str) -> bool:
        ok = await self.release_script(keys=[key])
        return bool(ok)

    async def _refresh_ttl_loop(self, key: str, stop_event: asyncio.Event) -> None:
        """Periodically check if the event's flag is not set, and extend the TTL of the key."""
        while True:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.refresh_interval)
            except TimeoutError:
                await self.redis.expire(key, self.ttl)
            else:
                return

    async def wrap_stream(self, stream: AsyncIterator[bytes], key: str) -> AsyncGenerator[Any]:
        """Hold the slot and prevent key expiry for the whole stream."""
        stop_event = asyncio.Event()
        watchdog = asyncio.create_task(self._refresh_ttl_loop(key, stop_event))
        try:
            async for chunk in stream:
                yield chunk
        finally:
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog
            await self.release(key)


async def _run_with_concurrency_limit(
    key: str,
    limit: int,
    error_message: str,
    view_func: AsyncView,
    request: ASGIRequest,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> ViewResult:
    """Acquire a slot for concurrent requests, run the view, then release the slot.

    The slot is released as soon as a non-streaming response is produced, or kept
    for the full duration of a streaming response. On an exception the slot is
    always released before the error propagates.
    """
    limiter = get_limiter()
    if not await limiter.acquire(key, limit):
        return error_response(error_message, status=429, headers={"Retry-After": "10"})

    try:
        response: ViewResult = await view_func(request, *args, **kwargs)
    except Exception:
        # Make sure we release the concurrent requests slot if there is an error.
        await limiter.release(key)
        raise

    if isinstance(response, StreamingHttpResponse):
        # Keep the slot held for the full stream duration.
        streaming_content = cast("AsyncIterator[bytes]", response.streaming_content)
        response.streaming_content = limiter.wrap_stream(streaming_content, key)
    else:
        # Response is fully generated; release now.
        await limiter.release(key)

    return response


def check_token_concurrency_limits(view_func: AsyncView) -> AsyncView:
    """Check concurrent requests limits per token."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_token_concurrency_limits used without token_authenticated decorator")
            return error_response("Internal server error", status=500)

        if not settings.AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED:
            return await view_func(request, *args, **kwargs)

        # Check concurrent requests limit per token, using a Redis counter
        limit = await sync_to_async(token_concurrency_limit)(token)
        return await _run_with_concurrency_limit(
            token_concurrency_key(token),
            limit,
            f"Token {token.name} sent too many concurrent requests",
            view_func,
            request,
            args,
            kwargs,
        )

    return wrapper


def check_model_concurrency_limits(view_func: AsyncView) -> AsyncView:
    """Check concurrent requests limits per model."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_model_concurrency_limits used without token_authenticated decorator")
            return error_response("Internal server error", status=500)

        if not settings.AQUEDUCT_MODEL_CONCURRENT_REQUESTS_LIMIT_ENABLED:
            return await view_func(request, *args, **kwargs)

        model = (kwargs.get("pydantic_model") or {}).get("model")
        if not model:
            return await view_func(request, *args, **kwargs)

        return await _run_with_concurrency_limit(
            model_concurrency_key(model),
            model_concurrency_limit(model),
            f"Too many concurrent requests for model {model}",
            view_func,
            request,
            args,
            kwargs,
        )

    return wrapper


def check_rate_limits(view_func: AsyncView) -> AsyncView:
    """Check token's rate limits, if enabled."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_rate_limits decorator used without token_authenticated decorator")
            return error_response("Internal server error", status=500)

        limits = await sync_to_async(token.get_limits)()
        log.debug("Rate limits for Token %r (ID: %s): %s", token.name, token.id, limits)

        if settings.AQUEDUCT_RATE_LIMIT_ENABLED and limits.has_any_rate_limit():
            model = (kwargs.get("pydantic_model") or {}).get("model")
            try:
                allowed, exceeded = await sync_to_async(check_and_reserve)(limits, token.id, model)
            except Exception as e:
                log.exception("Error checking rate limits for Token %r: %s", token.name, e)
                return error_response("Internal gateway error checking rate limits", status=500)

            if not allowed:
                messages = (
                    f"{message} (Retry after {seconds_left}s)" for message, seconds_left in exceeded
                )
                error_message = "Rate limit exceeded. " + ", ".join(messages) + "."
                log.warning(
                    "Rate limit exceeded for Token %r (ID: %s). Details: %s",
                    token.name,
                    token.id,
                    error_message,
                )
                # Wait until every exceeded bucket has ended
                retry_after = max(seconds_left for _message, seconds_left in exceeded)
                # Return 429 Too Many Requests
                return error_response(
                    error_message, status=429, headers={"Retry-After": str(retry_after)}
                )

        return await view_func(request, *args, **kwargs)

    return wrapper
