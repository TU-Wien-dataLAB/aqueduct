import logging
import time
import typing
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import cache as memoize
from typing import Any

import httpx2
import litellm
import openai
from django.core.cache import cache
from django.core.handlers.asgi import ASGIRequest
from openai import AsyncStream
from openai.types.responses import ResponseCreatedEvent, ResponseStreamEvent

from gateway.config import get_openai_client, get_router
from gateway.decorators.response_cache import register_response_in_cache

log = logging.getLogger("aqueduct")


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


@memoize
def _declared_fields(params_type: Any) -> frozenset[str]:
    members = typing.get_args(params_type) or (params_type,)
    return frozenset().union(*(m.__required_keys__ | m.__optional_keys__ for m in members))


def router_kwargs(body: Mapping[str, object], params_type: Any) -> dict[str, Any]:
    """Build LiteLLM router keyword arguments from a validated request body.

    LiteLLM takes routing settings and credentials (`api_base`, `api_key`, `fallbacks`,
    `mock_response`, ...) as keyword arguments next to the request parameters, so a body key
    passed as a keyword can redirect the request or the upstream key. Only fields declared by
    `params_type` (an OpenAI request TypedDict, or a union of them) and the gateway's `timeout`
    are passed as keywords; every other field goes to the upstream in `extra_body`.
    Nested client extras cannot override these validated fields in the upstream JSON.
    """
    allowed = _declared_fields(params_type) | {"timeout"}
    kwargs = {key: value for key, value in body.items() if key in allowed}
    extra_body = {key: value for key, value in body.items() if key not in allowed}
    client_extra_body = extra_body.pop("extra_body", None)
    if isinstance(client_extra_body, dict):
        # The upstream SDK merges extra_body over the request, including its checked model.
        client_extra_body = {
            key: value for key, value in client_extra_body.items() if key not in allowed
        }
        extra_body = {**client_extra_body, **extra_body}
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


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
            response=httpx2.Response(
                request=httpx2.Request(method=request.method, url=request.build_absolute_uri()),
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
            response=httpx2.Response(
                request=httpx2.Request(method=request.method, url=request.build_absolute_uri()),
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
