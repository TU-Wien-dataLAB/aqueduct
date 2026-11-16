import asyncio
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

from gateway.config import (
    get_model_max_parallel_requests,
    get_model_rpm,
    get_router_config,
    resolve_model,
)
from gateway.decorators.types import AsyncView, ViewResult
from gateway.model_rate_limiting import check_model_limit
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
    resolved = resolve_model(model_name)
    name = (
        resolved.accounting_name if resolved.accounting_name is not None else resolved.routing_name
    )
    return f"model:concurrency:{name}:leases"


def model_concurrency_limit(model: str) -> int:
    """Returns the concurrent requests limit for a specific model.

    Uses the configured model or router limit, falling back to the settings default.
    """
    configured = get_model_max_parallel_requests(model)
    if configured is not None:
        return configured
    return settings.AQUEDUCT_DEFAULT_MODEL_CONCURRENT_REQUESTS_LIMIT


class RedisConcurrencyLimiter:
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.acquire_script = redis_client.register_script(ACQUIRE_LUA)
        self.release_script = redis_client.register_script(RELEASE_LUA)
        self.refresh_script = redis_client.register_script(REFRESH_LUA)
        self.ttl = settings.AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_TTL_SECONDS

    async def acquire(self, key: str, limit: int) -> str | None:
        """Acquire a concurrent requests slot, returning its lease ID or None if over limit."""
        lease = uuid4().hex
        ok = await self.acquire_script(keys=[key], args=[limit, self.ttl, lease])
        return lease if ok else None

    async def refresh(self, key: str, lease: str) -> bool:
        """Renew only the owned, unexpired lease after stream progress."""
        ok = await self.refresh_script(keys=[key], args=[lease, self.ttl])
        return bool(ok)

    async def release(self, key: str, lease: str) -> bool:
        """Release the concurrent requests slot for the given lease ID."""
        ok = await self.release_script(keys=[key], args=[lease])
        return bool(ok)


class _LeaseLifetime:
    """Hold a reservation until completion or until stream progress stops."""

    def __init__(self, limiter: RedisConcurrencyLimiter, key: str, lease: str):
        self._limiter = limiter
        self._key = key
        self._lease = lease
        self._loop = asyncio.get_running_loop()
        self._release_task: asyncio.Task[bool] | None = None
        self._streaming = False

    async def __aenter__(self) -> "_LeaseLifetime":
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        # Release on errors/cancellation and after non-streaming responses.
        if exc_type is not None or not self._streaming:
            await self.aclose()

    @staticmethod
    def _observe_release(task: asyncio.Task[bool]) -> None:
        if task.cancelled():
            return
        if task.exception() is not None:
            log.error("Error releasing concurrency lease", exc_info=task.exception())

    def _start_release(self) -> asyncio.Task[bool]:
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._limiter.release(self._key, self._lease))
            self._release_task.add_done_callback(self._observe_release)
        return self._release_task

    async def aclose(self) -> None:
        """Release once without masking an upstream error or skipping cancellation."""
        task = self._start_release()
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                cancelled = True
            except Exception:
                # The release callback logs the failure; cleanup is best effort.
                break
        if task.cancelled():
            raise asyncio.CancelledError
        if cancelled:
            raise asyncio.CancelledError

    def _close(self) -> None:
        # Django may close a response without ever starting its async iterator.
        # close() can run in a worker thread, so start release on the owning loop.
        self._loop.call_soon_threadsafe(self._start_release)

    async def _wrap_stream(self, stream: AsyncIterator[bytes]) -> AsyncGenerator[bytes]:
        """Renew before forwarding non-empty chunks; stop if renewal fails."""
        try:
            async for chunk in stream:
                if chunk:
                    renewed = await self._limiter.refresh(self._key, self._lease)
                    if not renewed:
                        return
                yield chunk
        finally:
            await self.aclose()

    def wrap_response(self, response: StreamingHttpResponse) -> None:
        # Keep the reservation until stream completion, failure, or close.
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


def check_model_rate_limits(view_func: AsyncView) -> AsyncView:
    """Reserve one request from the shared public-model minute budget."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_model_rate_limits used without token_authenticated decorator")
            return error_response("Internal server error", status=500)

        model = (kwargs.get("pydantic_model") or {}).get("model")
        if not model:
            return await view_func(request, *args, **kwargs)

        try:
            model = resolve_model(model).accounting_name
            if model is None and any(
                entry.get("model_info", {}).get("rpm") is not None
                for entry in get_router_config().get("model_list", [])
            ):
                return error_response(
                    "Unknown public model. Use a configured model name, its Aqueduct alias, "
                    "or a configured deployment ID.",
                    param="model",
                    status=400,
                )
            rpm = get_model_rpm(model) if model is not None else None
            if model is not None and rpm is not None:
                # Excluded tokens must not consume capacity shared by all users.
                if await sync_to_async(token.model_excluded)(model):
                    return error_response("Model not found!", status=404)
                allowed, retry_after = await sync_to_async(check_model_limit)(model, rpm)
                if not allowed:
                    return error_response(
                        f"Model request limit ({rpm}/min) exceeded for '{model}' "
                        f"(Retry after {retry_after}s).",
                        status=429,
                        headers={"Retry-After": str(retry_after)},
                    )
        except Exception:
            log.exception("Error checking model rate limits for Token %r", token.name)
            return error_response("Internal gateway error checking rate limits", status=500)

        return await view_func(request, *args, **kwargs)

    return wrapper


def check_token_rate_limits(view_func: AsyncView) -> AsyncView:
    """Check token's rate limits, if enabled."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_token_rate_limits decorator used without token_authenticated decorator"
            )
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
