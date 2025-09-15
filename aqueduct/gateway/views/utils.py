import json
import time
from typing import AsyncGenerator, Any
from contextlib import contextmanager

from django.core.cache import cache

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm import TextCompletionStreamWrapper
from openai import AsyncStream

from management.models import Request, Usage


def _usage_from_bytes(content: bytes) -> Usage:
    try:
        usage_dict = json.loads(content).get('usage', None)
        if isinstance(usage_dict, dict):
            return Usage(
                input_tokens=usage_dict.get('prompt_tokens', 0),
                output_tokens=usage_dict.get('completion_tokens', 0)
            )
        else:
            return Usage(input_tokens=0, output_tokens=0)
    except json.JSONDecodeError:
        return Usage(input_tokens=0, output_tokens=0)


def _openai_stream(stream: CustomStreamWrapper | TextCompletionStreamWrapper | AsyncStream, request_log: Request) -> \
        AsyncGenerator[str, Any]:
    start_time = time.monotonic()

    async def _stream():
        token_usage = Usage(0, 0)
        async for chunk in stream:
            chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            token_usage += _usage_from_bytes(chunk_str.encode('utf-8'))
            try:
                yield f"data: {chunk_str}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        request_log.token_usage = token_usage
        request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
        await request_log.asave()
        # Streaming is done, yield the [DONE] chunk
        yield f"data: [DONE]\n\n"

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
