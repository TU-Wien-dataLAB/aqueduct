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
    return cast("aioredis.Redis", async_redis)


@lru_cache(maxsize=1)
def get_limiter() -> "RedisConcurrencyLimiter":
    return RedisConcurrencyLimiter(get_redis_client())


def concurrency_key(token: "Token") -> str:
    return f"user:concurrency:{token.id}"


def concurrency_limit_for(token: "Token") -> int:
    """Returns concurrent requests limit for the token."""
    if token.concurrent_requests_limit is not None:
        # Short-circuit if the limit is set on the token to avoid db queries
        return token.concurrent_requests_limit
    return token.get_limits().concurrent_requests_limit


class RedisConcurrencyLimiter:
    def __init__(self, redis_client: aioredis.Redis, refresh_interval: int = 30):
        self.redis = redis_client
        self.acquire_script = redis_client.register_script(ACQUIRE_LUA)
        self.release_script = redis_client.register_script(RELEASE_LUA)
        self.ttl = settings.AQUEDUCT_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS
        # Refresh interval has to be safely smaller than ttl; otherwise slots of long-running
        # stream requests may expire before the streaming is over!
        self.refresh_interval = min(refresh_interval, self.ttl // 2)

    async def acquire(self, token: "Token") -> bool:
        limit = concurrency_limit_for(token)
        ok = await self.acquire_script(keys=[concurrency_key(token)], args=[limit, self.ttl])
        return bool(ok)

    async def release(self, token: "Token") -> bool:
        ok = await self.release_script(keys=[concurrency_key(token)])
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

    async def wrap_stream(
        self, stream: AsyncIterator[bytes], token: "Token"
    ) -> AsyncGenerator[Any]:
        """Hold the slot and prevent key expiry for the whole stream."""
        key = concurrency_key(token)
        stop_event = asyncio.Event()
        watchdog = asyncio.create_task(self._refresh_ttl_loop(key, stop_event))
        try:
            async for chunk in stream:
                yield chunk
        finally:
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog
            await self.release(token)


def check_concurrency_limits(view_func: AsyncView) -> AsyncView:
    """Check concurrency limits per token."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_concurrency_limits decorator used without token_authenticated decorator"
            )
            return error_response("Internal server error", status=500)

        if not settings.AQUEDUCT_CONCURRENT_REQUESTS_LIMIT_ENABLED:
            return await view_func(request, *args, **kwargs)

        # Check concurrent requests limit per token, using a Redis counter
        limiter = get_limiter()
        if not await limiter.acquire(token):
            return error_response(
                f"Token {token.name} sent too many concurrent requests",
                status=429,
                headers={"Retry-After": "10"},
            )

        try:
            response: ViewResult = await view_func(request, *args, **kwargs)
        except Exception:
            # Make sure we release the concurrent requests slot if there is an error.
            await limiter.release(token)
            raise

        if isinstance(response, StreamingHttpResponse):
            # Keep the slot held for the full stream duration.
            streaming_content = cast("AsyncIterator[bytes]", response.streaming_content)
            response.streaming_content = limiter.wrap_stream(streaming_content, token)
        else:
            # Response is fully generated; release now.
            await limiter.release(token)

        return response

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
