import json
import logging
import time
from contextlib import contextmanager
from typing import Any, AsyncGenerator

import httpx
import litellm
import openai
from django.core.cache import cache, caches
from django.core.handlers.asgi import ASGIRequest
from litellm import TextCompletionStreamWrapper
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from openai import AsyncStream

from gateway.config import get_openai_client, get_router
from management.models import Request, Usage

log = logging.getLogger("aqueduct")


def _get_token_usage(content: bytes | dict) -> Usage:
    """Retrieves token usage information from the response content.

    Note that if the response data does not match the expected format, or does
    not contain the usage information, the returned token usage will be wrong,
    i.e. set to 0.

    Args:
        content: The response content, either as a dict or as bytes.
    Returns:
        The :class:`Usage` object with the used input and output token counts.
    """

    if isinstance(content, dict):
        data = content
    else:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return Usage(input_tokens=0, output_tokens=0)

    # Handle responses API format (top-level usage or in response field)
    usage_dict = data.get("usage", None)
    if not usage_dict and "response" in data:
        usage_dict = data["response"].get("usage", None)
    if isinstance(usage_dict, dict):
        input_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens", 0)
        output_tokens = usage_dict.get("completion_tokens") or usage_dict.get("output_tokens", 0)
        return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
    else:
        return Usage(input_tokens=0, output_tokens=0)


def _openai_stream(
    stream: CustomStreamWrapper | TextCompletionStreamWrapper | AsyncStream, request_log: Request
) -> AsyncGenerator[str, Any]:
    start_time = time.monotonic()

    async def _stream():
        token_usage = Usage(0, 0)
        async for chunk in stream:
            chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)

            # Extract token usage from this chunk
            chunk_usage = _get_token_usage(chunk_str.encode("utf-8"))

            # Only update if we got actual usage data (non-zero tokens)
            if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                token_usage = chunk_usage

            try:
                yield f"data: {chunk_str}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        request_log.token_usage = token_usage
        request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
        await request_log.asave()
        # Streaming is done, yield the [DONE] chunk
        yield "data: [DONE]\n\n"

    return _stream()


@contextmanager
def cache_lock(lock_id, ttl: int):
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
    """Check if a value is in a list of allowed values or if it matches a wildcard pattern within these values."""
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
    try:
        client: openai.AsyncClient = get_openai_client(model)
    except ValueError:
        raise openai.NotFoundError(
            message=f"Incompatible model '{model}'!",
            response=httpx.Response(
                request=httpx.Request(method=request.method, url=request.build_absolute_uri()),
                status_code=404,
            ),
            body=None,
        )

    router = get_router()
    deployment: litellm.Deployment = router.get_deployment(model_id=model)

    model_relay, provider, _, _ = litellm.get_llm_provider(deployment.litellm_params.model)
    return client, model_relay


class ResponseRegistrationWrapper:
    """Wraps streaming content to register response on first chunk."""

    def __init__(self, streaming_content, model: str, email: str):
        self.streaming_content = streaming_content
        self.model_name = model
        self.user_email = email
        self._registered = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Iterate through the streaming content
            if hasattr(self.streaming_content, "__anext__"):
                chunk = await self.streaming_content.__anext__()
            else:
                # Handle iterator-like objects
                chunk = next(self.streaming_content)

            if not self._registered and chunk:
                response_id = self.extract_response_id_from_chunk(chunk)
                if response_id:
                    register_response_in_cache(response_id, self.model_name, self.user_email)
                    self._registered = True

            return chunk
        except StopAsyncIteration:
            raise
        except StopIteration:
            raise StopAsyncIteration

    @staticmethod
    def extract_response_id_from_chunk(chunk: bytes) -> str | None:
        """Extract response ID from SSE chunk containing 'response.created' event."""
        try:
            chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
            # Parse SSE format: "data: {json}"
            for line in chunk_str.split("\n"):
                if line.startswith("data: ") and "response.created" in line:
                    json_data = json.loads(line[6:])  # Remove "data: " prefix
                    if json_data.get("type") == "response.created":
                        return json_data.get("response", {}).get("id")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        return None


def register_response_in_cache(response_id: str | None, model: str, email: str):
    """Registers a response in the cache for later retrieval."""
    if not response_id:
        log.warning(f"Missing response data: id={response_id}, model={model}")
        raise ValueError("Missing response_id")

    cache_key = f"response:{response_id}"
    cache_value = {"model": model, "email": email}

    response_cache = caches["default"]
    response_cache.set(cache_key, cache_value, timeout=3600)
    log.debug(f"Registered response {response_id} for user {email} with model {model}")


def get_response_from_cache(response_id: str) -> dict | None:
    """Retrieves a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    return response_cache.get(cache_key)
