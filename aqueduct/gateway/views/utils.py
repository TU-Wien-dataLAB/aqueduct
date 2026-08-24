import logging
import time
from collections.abc import AsyncIterator, Callable, Generator
from contextlib import contextmanager
from typing import Any

import httpx
import litellm
import openai
from django.conf import settings
from django.core.cache import cache, caches
from django.core.handlers.asgi import ASGIRequest
from django.http.response import ResponseHeaders
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import Usage as UsageModel
from openai import AsyncStream
from openai.types.responses import ResponseCreatedEvent, ResponseStreamEvent
from pydantic import BaseModel

from gateway.config import get_openai_client, get_router
from management.models import Request, Usage

log = logging.getLogger("aqueduct")


class RawJsonResponse:
    """A wrapper for data that can be turned into a JSONResponse."""

    def __init__(self, data: dict[str, Any], **kwargs: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("RawJsonResponse data has to be a dict")

        self.content = data
        self.kwargs = kwargs or {}
        # The following mimics the BaseHttpResponse behaviour (argument called "status"
        # is assigned to the "status_code" attribute)
        self.status_code = self.kwargs.get("status", 200)
        # Just to be on the safe side, make header keys case-insensitive:
        self.headers = ResponseHeaders(self.kwargs.get("headers", {}))
        self.content_type = self.kwargs.get("content_type", "application/json")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} status_code={self.status_code}>"


class RawStreamingResponse:
    """A wrapper for streaming data that can be turned into a StreamingHttpResponse."""

    def __init__(
        self,
        streaming_content: AsyncIterator[Any],
        request_log: Request,  # TODO: for mcp requests, there's no request log!
        transforms: list[Callable[[ModelResponseStream], ModelResponseStream]]
        | None = None,  # TODO: fix types
        **kwargs: Any,
    ) -> None:
        if not isinstance(streaming_content, AsyncIterator):
            raise TypeError("RawStreamResponse streaming_content has to be async iterable")

        self.streaming_content = streaming_content
        self.request_log = request_log
        self.transforms = transforms or []
        self.kwargs = kwargs or {}
        # The following mimics the BaseHttpResponse behaviour (argument called "status"
        # is assigned to the "status_code" attribute)
        self.status_code = self.kwargs.get("status", 200)
        # Just to be on the safe side, make header keys case-insensitive:
        self.headers = ResponseHeaders(self.kwargs.get("headers", {}))
        self.content_type = self.kwargs.get("content_type", "text/event-stream")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} status_code={self.status_code}>"


def get_token_usage(data: dict[str, Any] | BaseModel) -> Usage:
    """Retrieves token usage information from the raw response content.

    Note that if the response data does not match the expected format, or does
    not contain the usage information, the returned token usage will be wrong,
    i.e. set to 0.

    Args:
        data: The raw response content (or content's chunk for streaming responses)
          as a dict or BaseModel subclass (for RawStreamingResponses).
    Returns:
        The :class:`Usage` object with the used input and output token counts.
    """
    # Handle responses API format (top-level usage or in response field)
    if isinstance(data, (dict, ModelResponseStream)):
        # LiteLLM models implement `.get()` method, but the OpenAI ones - don't.
        usage = data.get("usage")
    else:
        data = data.model_dump(exclude_none=True, exclude_unset=True)
        usage = data.get("usage")
    if not usage and "response" in data:
        usage = data["response"].get("usage")
    if isinstance(usage, (dict, UsageModel)):
        input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
        output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    return Usage(input_tokens=0, output_tokens=0)


@contextmanager
def cache_lock(lock_id: str, ttl: int) -> Generator[bool, None, None]:
    """
    Acquire a cache-based lock with key `lock_id`, and expiration `ttl` seconds.
    Yields True if the lock was acquired (cache.add succeeded), False otherwise.
    Ensures lock is only released if still within ttl window and owned by us.
    """
    timeout_at = time.monotonic() + ttl
    status = cache.add(lock_id, 0, ttl)
    try:
        yield status
    finally:
        if status and time.monotonic() < timeout_at:
            cache.delete(lock_id)


def in_wildcard(value: str | None, allowed_values: list[str]) -> bool:
    """Check if a value is in a list of allowed values or matches a wildcard pattern."""
    if value is None:
        return False

    valid = value in allowed_values
    if not valid:
        # Check wildcard port patterns (e.g., "http://localhost:*")
        for allowed in allowed_values:
            if allowed.endswith(":*"):
                base_origin = allowed[:-2]
                if value.startswith(base_origin + ":"):
                    return True
    return valid


def oai_client_from_body(model: str, request: ASGIRequest) -> tuple[openai.AsyncClient, str]:
    """Returns an OpenAI-compatible async client and provider-specific model name for proxying.
    Used when direct OpenAI SDK client is needed instead of LiteLLM router
    (e.g., Responses API, Batches API).
    """
    try:
        client: openai.AsyncClient = get_openai_client(model)
    except ValueError:
        log.exception("Incompatible model '%s'! Is model id set in router config?", model)
        raise openai.NotFoundError(
            message=f"Incompatible model '{model}'!",
            response=httpx.Response(
                request=httpx.Request(method=request.method, url=request.build_absolute_uri()),
                status_code=404,
            ),
            body=None,
        ) from None

    router = get_router()
    deployment: litellm.Deployment | None = router.get_deployment(model_id=model)

    if deployment is None:
        log.error("Model '%s' not found in router deployments", model)
        raise openai.NotFoundError(
            message=f"Model '{model}' not found!",
            response=httpx.Response(
                request=httpx.Request(method=request.method, url=request.build_absolute_uri()),
                status_code=404,
            ),
            body=None,
        )

    model_relay, _provider, _, _ = litellm.get_llm_provider(deployment.litellm_params.model)
    return client, model_relay


class ResponseRegistrationWrapper:
    """Wraps streaming content to register response on first chunk."""

    def __init__(self, streaming_content: AsyncStream[ResponseStreamEvent], model: str, email: str):
        self.streaming_content = streaming_content
        self.model_name = model
        self.user_email = email
        self._registered = False

    def __aiter__(self) -> "ResponseRegistrationWrapper":
        return self

    async def __anext__(self) -> ResponseStreamEvent:
        chunk: ResponseStreamEvent = await self.streaming_content.__anext__()
        if (
            not self._registered
            and isinstance(chunk, ResponseCreatedEvent)
            and chunk.type == "response.created"
        ):
            response_id: str | None = chunk.response.id
            if response_id:
                register_response_in_cache(response_id, self.model_name, self.user_email)
                self._registered = True
        return chunk


def register_response_in_cache(response_id: str | None, model: str, email: str) -> None:
    """Registers a response in the cache for later retrieval."""
    if not response_id:
        log.warning("Missing response data: id=%s, model=%s", response_id, model)
        raise ValueError("Missing response_id")

    cache_key = f"response:{response_id}"
    cache_value = {"model": model, "email": email}

    response_cache = caches["default"]
    response_cache.set(cache_key, cache_value, timeout=settings.RESPONSES_API_TTL_SECONDS)
    log.debug("Registered response %s for user %s with model %s", response_id, email, model)


def get_response_from_cache(response_id: str) -> dict[str, Any] | None:
    """Retrieves a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    result: dict[str, Any] | None = response_cache.get(cache_key)
    return result


def delete_response_from_cache(response_id: str) -> None:
    """Deletes a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    response_cache.delete(cache_key)
