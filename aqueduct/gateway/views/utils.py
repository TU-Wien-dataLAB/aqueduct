import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator
from contextlib import contextmanager
from functools import reduce
from typing import Any, Literal, TypeVar

import httpx
import litellm
import openai
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.cache import cache, caches
from django.core.handlers.asgi import ASGIRequest
from django.core.serializers.json import DjangoJSONEncoder
from django.http.response import HttpResponseBase, StreamingHttpResponse
from litellm.types.utils import (
    EmbeddingResponse,
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)
from litellm.types.utils import Usage as UsageModel
from mcp.types import JSONRPCMessage
from openai import AsyncStream
from openai.types.responses import ResponseCreatedEvent, ResponseStreamEvent
from pydantic import BaseModel

from gateway.config import get_openai_client, get_router
from gateway.rate_limiting import record_token_usage
from management.models import Request, Usage

log = logging.getLogger("aqueduct")

T = TypeVar("T", bound=ModelResponseStream | JSONRPCMessage)


class RawJsonResponse(HttpResponseBase):
    """Mimics JSONResponse behaviour, but dumps data to JSON lazily - only on access."""

    streaming = False

    def __init__(self, data: dict[str, Any] | BaseModel, **kwargs: Any) -> None:
        if not isinstance(data, (dict, BaseModel)):
            raise TypeError("RawJsonResponse data has to be a dict or a pydantic BaseModel")

        # ``data`` is the original object passed when creating the response instance
        self.data = data
        # ``_content`` stores the data dumped to a string
        self._content: bytes | None = None
        kwargs.setdefault("content_type", "application/json")
        super().__init__(**kwargs)

    @property
    def content(self) -> bytes:
        if self._content is None:
            self._content = self._dump_data()
        return self._content

    def _dump_data(self) -> bytes:
        """Serialize ``self.data`` to JSON and return the result as bytes."""
        _content = {}
        if isinstance(self.data, BaseModel):
            _content = self.data.model_dump(exclude_none=True, exclude_unset=True, mode="json")
        else:
            for k, v in self.data.items():
                if isinstance(v, BaseModel):
                    # Data can be a dict containing models as values
                    _content[k] = v.model_dump(exclude_none=True, exclude_unset=True, mode="json")
                elif isinstance(v, (list, tuple)) and any(
                    isinstance(item, BaseModel) for item in v
                ):
                    # Data can be a dict containing a list of models
                    _content[k] = [
                        item.model_dump(exclude_none=True, exclude_unset=True, mode="json")
                        for item in v
                    ]
                else:
                    _content[k] = v

        return self.make_bytes(json.dumps(_content, cls=DjangoJSONEncoder))

    @property
    def text(self) -> str:
        return self.content.decode(self.charset or "utf-8")


def _apply_transforms(chunk: T, transforms: list[Callable[[T], T]]) -> T:
    return reduce(lambda obj, tr: tr(obj), transforms, chunk)


class RawStreamingResponse(StreamingHttpResponse):
    """A wrapper for streaming data that can be turned into a StreamingHttpResponse."""

    def __init__(
        self,
        streaming_content: AsyncIterator[Any],
        request_log: Request | None,
        transforms: list[Callable[[T], T]] | None = None,
        *,
        mode: Literal["openai", "mcp"] = "openai",
        **kwargs: Any,
    ) -> None:
        if not isinstance(streaming_content, AsyncIterator):
            raise TypeError("RawStreamResponse streaming_content has to be async iterable")

        self.streaming_content = streaming_content
        self.request_log = request_log
        self.transforms = transforms or []
        self.mode = mode
        kwargs.setdefault("content_type", "text/event-stream")
        super().__init__(streaming_content=(), **kwargs)
        self.streaming_content = self._iter_stream(streaming_content)

    async def _iter_stream(self, streaming_content: AsyncIterator[Any]) -> AsyncGenerator[bytes]:
        """Post-process streaming response chunks with transforms, and log OpenAI responses.

        Note: MCP streaming responses do not have `request_log` attached;
        usage tokens and response time are only logged for OpenAI responses.
        Also, OpenAI responses require the last yielded chunk to be b"data: [DONE]\n\n",
        which however is not MCP-compliant.
        """
        token_usage = Usage(0, 0)
        start_time = time.monotonic()
        is_openai = self.mode == "openai"
        log.debug(
            "%r stream. Applying the following transforms to each chunk: %s",
            self.mode,
            self.transforms,
        )

        if is_openai and self.request_log is None:
            raise ValueError(f"Missing request_log for an OpenAI streaming response: {self}!")

        async for raw_chunk in streaming_content:
            chunk = _apply_transforms(raw_chunk, self.transforms)

            if is_openai:
                chunk_usage = get_token_usage(chunk)
                if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                    token_usage = chunk_usage
                chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            else:
                chunk_str = chunk.model_dump_json(exclude_none=True)

            yield f"data: {chunk_str}\n\n".encode()

        if is_openai:
            if self.request_log is None:
                raise ValueError(f"Missing request_log for an OpenAI streaming response: {self}!")
            self.request_log.token_usage = token_usage
            self.request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
            await self.request_log.asave()
            # Record token usage into the rate-limit buckets. Streaming requests
            # defer recording to here (stream end) since token usage is only known
            # once the upstream stream completes.
            await sync_to_async(record_token_usage)(
                self.request_log.token_id, self.request_log.token_usage
            )

            yield b"data: [DONE]\n\n"


def get_token_usage(data: dict[str, Any] | BaseModel) -> Usage:
    """Retrieves token usage information from the raw response data.

    Note that if the response data does not match the expected format, or does
    not contain the usage information, the returned token usage will be wrong,
    i.e. set to 0.

    Args:
        data: The raw response data (or content's chunk for streaming responses),
          expected to be a dict or BaseModel subclass.
    Returns:
        The :class:`Usage` object with the used input and output token counts.
    """
    if isinstance(
        data, (dict, ModelResponse, ModelResponseStream, EmbeddingResponse, TextCompletionResponse)
    ):
        # LiteLLM models implement `.get()` method, but the OpenAI ones - don't.
        usage = data.get("usage")
        if isinstance(usage, (dict, UsageModel)):
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
            return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    else:
        # Handle responses API format (top-level usage or in response field)
        usage = getattr(data, "usage", None)
        if not usage and hasattr(data, "response"):
            usage = getattr(data.response, "usage", None)
        if usage:
            try:
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
            except AttributeError:
                input_tokens = output_tokens = 0
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
        if not self._registered and isinstance(chunk, ResponseCreatedEvent):
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
