import json
import logging
import time
from collections.abc import AsyncGenerator, Callable
from functools import reduce
from typing import Any

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from pydantic import BaseModel

from gateway.views.utils import RawJsonResponse, RawStreamingResponse, get_token_usage
from management.models import Usage

log = logging.getLogger("aqueduct")


def _dump_to_json(chunk: BaseModel | dict[str, Any]) -> str:
    """Convert the content chunk to a JSON string (suitable for StreamingHttpResponse)"""
    if isinstance(chunk, BaseModel):
        return chunk.model_dump_json(exclude_none=True, exclude_unset=True)
    if isinstance(chunk, dict):
        return json.dumps(chunk)

    raise TypeError(f"Received unexpected streaming chunk type, {type(chunk)}!")


async def _openai_stream(response: RawStreamingResponse) -> AsyncGenerator[str, None]:
    """Process streaming response with transforms and log token usage."""
    token_usage = Usage(0, 0)
    start_time = time.monotonic()
    request_log = response.request_log

    log.debug("Applying the following transforms to each chunk: %s", response.transforms)
    async for raw_chunk in response.streaming_content:
        # Apply all registered transforms
        chunk = reduce(lambda obj, tr: tr(obj), response.transforms, raw_chunk)

        if request_log is not None:
            # Retrieve token usage
            chunk_usage = get_token_usage(chunk)
            if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
                token_usage = chunk_usage

        chunk_str = _dump_to_json(chunk)

        try:
            yield f"data: {chunk_str}\n\n"
        except Exception as e:
            yield f"data: {e!s}\n\n"

    if request_log is not None:
        end_time = time.monotonic()
        request_log.token_usage = token_usage
        request_log.response_time_ms = int((end_time - start_time) * 1000)
        await request_log.asave()

    # Streaming is done, yield the [DONE] chunk
    yield "data: [DONE]\n\n"


class HttpResponseMiddleware:
    """
    Transform raw responses from the gateway views into valid HTTPResponses.

    This middleware should be applied after all data post-processing
    middleware.
    """

    def __init__(
        self, get_response: Callable[[ASGIRequest], JsonResponse | StreamingHttpResponse]
    ) -> None:
        self.get_response = get_response

    def __call__(self, request: ASGIRequest) -> JsonResponse | StreamingHttpResponse:
        """Transform a raw response from a gateway view into a valid HTTPResponse.

        If the response is not an instance of ``RawJsonResponse``
        or ``RawStreamingResponse``, it is returned unchanged.
        """
        response = self.get_response(request)

        if isinstance(response, RawJsonResponse):
            # Merge headers from response.headers (may have been modified after init)
            kwargs = response.kwargs.copy()
            kwargs["headers"] = dict(response.headers)
            return JsonResponse(response.content, **kwargs)

        if isinstance(response, RawStreamingResponse):
            # Merge headers from response.headers (may have been modified after init)
            kwargs = response.kwargs.copy()
            kwargs["headers"] = dict(response.headers)

            if request.path.startswith("/mcp-servers/"):  # what about is_initialize?
                # MCP responses need special treatment... but this doesn't work still (TODO!)
                async def apply_transforms() -> AsyncGenerator[str]:
                    async for raw_chunk in response.streaming_content:
                        chunk = reduce(lambda obj, tr: tr(obj), response.transforms, raw_chunk)
                        chunk_str = json.dumps(chunk)
                        yield f"data: {chunk_str}\n\n"

                streaming_content = apply_transforms()
            else:
                # for non-MCP responses
                streaming_content = _openai_stream(response)

            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        return response
