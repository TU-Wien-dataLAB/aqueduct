"""Utility helpers for gateway views.

The raw response wrappers (``RawJsonResponse``, ``RawStreamingResponse``) and
``get_token_usage`` live in ``gateway.response_type`` so they can be shared with
the ``gateway.decorators`` package without a circular import;
they are re-exported here to keep existing ``gateway.views.utils`` imports
working. ``cache_lock``, ``oai_client_from_body`` and ``ResponseRegistrationWrapper``
remain view-specific and are defined below.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager

import httpx
import litellm
import openai
from django.core.cache import cache
from django.core.handlers.asgi import ASGIRequest
from openai import AsyncStream
from openai.types.responses import ResponseCreatedEvent, ResponseStreamEvent

from gateway.config import get_openai_client, get_router
from gateway.decorators.response_cache import register_response_in_cache
from gateway.response_type import RawJsonResponse, RawStreamingResponse, get_token_usage

log = logging.getLogger("aqueduct")

__all__ = [
    "RawJsonResponse",
    "RawStreamingResponse",
    "ResponseRegistrationWrapper",
    "cache_lock",
    "get_token_usage",
    "oai_client_from_body",
]


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
        if not self._registered and isinstance(chunk, ResponseCreatedEvent):
            response_id: str | None = chunk.response.id
            if response_id:
                register_response_in_cache(response_id, self.model_name, self.user_email)
                self._registered = True
        return chunk
