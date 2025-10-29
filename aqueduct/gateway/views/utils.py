import json
import time
from contextlib import contextmanager
from typing import Any, AsyncGenerator

from django.core.cache import cache
from litellm import TextCompletionStreamWrapper
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from openai import AsyncStream

from management.models import Request, Usage


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
        usage_dict = content.get("usage", None)
    else:
        try:
            usage_dict = json.loads(content).get("usage", None)
        except json.JSONDecodeError:
            return Usage(input_tokens=0, output_tokens=0)

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
            token_usage += _get_token_usage(chunk_str.encode("utf-8"))
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
