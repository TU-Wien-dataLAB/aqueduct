import asyncio
import contextlib
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

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


# Lua: remove expired leases for the `KEYS[1]` key, then acquire a slot if below limit.
# ARGV[1] = limit, ARGV[2] = TTL, ARGV[3] = lease ID: each lease expires after this time
# in case the backend failed to call the release script for whatever reason (i.e. worker failure).
# Lease expiry uses Redis time, so healthy requests do not keep orphaned slots alive.
ACQUIRE_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then
    return 0
end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""


# Lua: release the lease for the `KEYS[1]` key, making sure other leases are not removed.
# ARGV[1] = lease ID.
RELEASE_LUA = """
return redis.call('ZREM', KEYS[1], ARGV[1])
"""


# Lua: extend the TTL of the lease for the `KEYS[1]` key, only if it has not expired.
# ARGV[1] = lease ID, ARGV[2] = TTL.
REFRESH_LUA = """
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local expiry = redis.call('ZSCORE', KEYS[1], ARGV[1])
if not expiry or tonumber(expiry) <= now then
    return 0
end
redis.call('ZADD', KEYS[1], 'XX', now + tonumber(ARGV[2]), ARGV[1])
redis.call('EXPIRE', KEYS[1], ARGV[2])
return 1
"""


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    return aioredis.Redis.from_url(settings.REDIS_URL, decode_responses=True)


@lru_cache(maxsize=1)
def get_limiter() -> "RedisConcurrencyLimiter":
    return RedisConcurrencyLimiter(get_redis_client())


# Separate from legacy string counters to avoid Redis WRONGTYPE errors. Drain
# counter-based workers before deployment: mixed versions use separate pools.
def token_concurrency_key(token: "Token") -> str:
    return f"token:concurrency:{token.id}:leases"


def token_concurrency_limit(token: "Token") -> int:
    """Returns concurrent requests limit for the token."""
    if token.concurrent_requests_limit is not None:
        # Short-circuit if the limit is set on the token to avoid db queries
        return token.concurrent_requests_limit
    return token.get_limits().concurrent_requests_limit


def model_concurrency_key(model_name: str) -> str:
    return f"model:concurrency:{resolve_model_alias(model_name)}:leases"


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
    def __init__(self, redis_client: aioredis.Redis, refresh_interval: float = 30):
        self.redis = redis_client
        self.acquire_script = redis_client.register_script(ACQUIRE_LUA)
        self.release_script = redis_client.register_script(RELEASE_LUA)
        self.refresh_script = redis_client.register_script(REFRESH_LUA)
        self.ttl = settings.AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS
        # Refresh interval has to be safely smaller than ttl; otherwise slots of long-running
        # stream requests may expire before the streaming is over!
        self.refresh_interval = min(refresh_interval, self.ttl / 2)

    async def acquire(self, key: str, limit: int) -> str | None:
        """Acquire a concurrent requests slot, returning its lease ID or None if over limit."""
        lease = uuid4().hex
        ok = await self.acquire_script(keys=[key], args=[limit, self.ttl, lease])
        return lease if ok else None

    async def release(self, key: str, lease: str) -> bool:
        """Release the concurrent requests slot for the given lease ID."""
        ok = await self.release_script(keys=[key], args=[lease])
        return bool(ok)


class _LeaseLifetime:
    """Hold the slot and prevent lease expiry for the whole request."""

    def __init__(self, limiter: RedisConcurrencyLimiter, key: str, lease: str):
        self._limiter = limiter
        self._key = key
        self._lease = lease
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._watchdog = asyncio.create_task(self._refresh_ttl_loop())
        self._streaming = False

    async def __aenter__(self) -> "_LeaseLifetime":
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        # Make sure we release the concurrent requests slot if there is an error
        # (including cancellation), or the non-streaming response is fully generated.
        if exc_type is not None or not self._streaming:
            await self._aclose()

    async def _refresh_ttl_loop(self) -> None:
        """Periodically check if the event's flag is not set, and extend the TTL of the lease.

        Release the concurrent requests slot when the loop stops.
        """
        try:
            while True:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._limiter.refresh_interval
                    )
                except TimeoutError:
                    if not await self._limiter.refresh_script(
                        keys=[self._key], args=[self._lease, self._limiter.ttl]
                    ):
                        return
                else:
                    return
        finally:
            await self._limiter.release(self._key, self._lease)

    async def _aclose(self) -> None:
        self._stop_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await self._watchdog

    def _close(self) -> None:
        # Django may close a response without ever starting its async iterator.
        # close() can run in a worker thread, so signal the owning event loop.
        self._loop.call_soon_threadsafe(self._stop_event.set)

    async def _wrap_stream(self, stream: AsyncIterator[bytes]) -> AsyncGenerator[bytes]:
        """Hold the slot and prevent lease expiry for the whole stream."""
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await self._aclose()

    def wrap_response(self, response: StreamingHttpResponse) -> None:
        # Keep the slot held for the full stream duration.
        streaming_content = cast("AsyncIterator[bytes]", response.streaming_content)
        response.streaming_content = self._wrap_stream(streaming_content)
        response._resource_closers.append(self._close)  # type: ignore[attr-defined]
        self._streaming = True


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
    lease = await limiter.acquire(key, limit)
    if lease is None:
        return error_response(error_message, status=429, headers={"Retry-After": "10"})

    async with _LeaseLifetime(limiter, key, lease) as lifetime:
        response: ViewResult = await view_func(request, *args, **kwargs)
        if isinstance(response, StreamingHttpResponse):
            lifetime.wrap_response(response)
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
