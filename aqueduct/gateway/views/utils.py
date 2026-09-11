import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import httpx
import litellm
import openai
from django.core.cache import cache
from django.core.handlers.asgi import ASGIRequest
from litellm.types.utils import (
    EmbeddingResponse,
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)
from litellm.types.utils import Usage as UsageModel
from openai import AsyncStream
from openai.types.responses import ResponseCreatedEvent, ResponseStreamEvent
from pydantic import BaseModel

from gateway.config import get_openai_client, get_router
from gateway.raw_response import (
    RawJsonResponse,
    RawStreamingResponse,
    delete_response_from_cache,
    get_response_from_cache,
    in_wildcard,
    register_response_in_cache,
)
from management.models import Usage

log = logging.getLogger("aqueduct")

__all__ = [
    "RawJsonResponse",
    "RawStreamingResponse",
    "delete_response_from_cache",
    "get_response_from_cache",
    "in_wildcard",
    "register_response_in_cache",
]


def get_token_usage(data: dict[str, Any] | BaseModel) -> Usage:
    """Retrieves token usage information from the raw response content.

    Note that if the response data does not match the expected format, or does
    not contain the usage information, the returned token usage will be wrong,
    i.e. set to 0.

    Args:
        data: The raw response content (or content's chunk for streaming responses),
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
