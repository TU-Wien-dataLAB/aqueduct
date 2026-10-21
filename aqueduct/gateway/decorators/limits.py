import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import redis.asyncio as aioredis
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.http.response import StreamingHttpResponse
from redis import Redis

from gateway.decorators.types import AsyncView, ViewResult
from gateway.rate_limiting import check_and_reserve, has_any_limit
from gateway.response_error import error_response

if TYPE_CHECKING:
    from django.contrib.auth import get_user_model

    from management.models import Token

    User = get_user_model()


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


redis_client = aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
acquire_script = redis_client.register_script(ACQUIRE_LUA)
release_script = redis_client.register_script(RELEASE_LUA)


def concurrency_key(user_id: int) -> str:
    return f"user:concurrency:{user_id}"


def concurrency_limit_for(user_id: int) -> int:
    """A static value for now. Allows to adjust the limits on the per-user basis, if needed."""
    return settings.AQUEDUCT_DEFAULT_CONCURRENT_REQUESTS_LIMIT


class RedisConcurrencyLimiter:
    def __init__(self, redis_client: Redis, refresh_interval: int = 30):
        self.redis = redis_client
        self.ttl = settings.AQUEDUCT_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS
        self.refresh_interval = refresh_interval

    async def acquire(self, user_id: int) -> bool:
        limit = concurrency_limit_for(user_id)
        ok = await acquire_script(keys=[concurrency_key(user_id)], args=[limit, self.ttl])
        return bool(ok)

    async def release(self, user_id: int) -> bool:
        ok = await release_script(keys=[concurrency_key(user_id)])
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

    async def wrap_stream(self, stream: AsyncIterator[bytes], user_id: int) -> AsyncGenerator[Any]:
        """Hold the slot and prevent key expiry for the whole stream."""
        key = concurrency_key(user_id)
        stop_event = asyncio.Event()
        watchdog = asyncio.create_task(self._refresh_ttl_loop(key, stop_event))
        try:
            async for chunk in stream:
                yield chunk
        finally:
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await watchdog
            await self.release(user_id)


limiter = RedisConcurrencyLimiter(redis_client)


def check_limits(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_limits decorator used without @token_authenticated decorator")
            return error_response("Internal server error", status=500)

        # 1. Check concurrent requests limit, using a Redis counter
        user_id = token.user.id
        if not await limiter.acquire(user_id):
            return error_response(
                "User sent too many concurrent requests",
                status=429,
                headers={"Retry-After": "30"},  # TODO: retry-after value???
            )

        # 2. Check rate limits
        limits = await sync_to_async(token.get_limit)()
        log.debug("Rate limits for Token %r (ID: %s): %s", token.name, token.id, limits)

        if settings.AQUEDUCT_RATE_LIMIT_ENABLED and has_any_limit(limits):
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

        try:
            response = await view_func(request, *args, **kwargs)
        except Exception:
            # Make sure we release the connection slot if there is an error.
            await limiter.release(user_id)
            raise

        if isinstance(response, StreamingHttpResponse):
            # Keep the slot held for the full stream duration.
            streaming_content = cast("AsyncIterator[bytes]", response.streaming_content)
            response.streaming_content = limiter.wrap_stream(streaming_content, user_id)
        else:
            # Response is fully generated; release now.
            await limiter.release(user_id)
        return response

    return wrapper
