import logging
import time
from collections.abc import AsyncGenerator, Callable
from functools import reduce
from typing import TypeVar

from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from litellm.types.utils import ModelResponseStream
from mcp.types import JSONRPCMessage
from pydantic import BaseModel

from gateway.views.utils import RawJsonResponse, RawStreamingResponse, get_token_usage
from management.models import Usage

log = logging.getLogger("aqueduct")


ViewResult = HttpResponse | StreamingHttpResponse | RawJsonResponse | RawStreamingResponse
T = TypeVar("T", bound=ModelResponseStream | JSONRPCMessage)


def _apply_transforms(chunk: T, transforms: list[Callable[[T], T]]) -> T:
    return reduce(lambda obj, tr: tr(obj), transforms, chunk)


async def _openai_stream(response: RawStreamingResponse) -> AsyncGenerator[str, None]:
    """Post-process streaming response chunks with transforms, and log token usage and request time.

    Note: the last yielded chunk, "data: [DONE]\n\n", is not MCP-compliant. Also,
    MCP streaming responses do not have `request_log` attached. Use ``_mcp_stream()``
    for post-processing of MCP streaming responses.
    """
    token_usage = Usage(0, 0)
    start_time = time.monotonic()
    request_log = response.request_log
    if request_log is None:
        raise ValueError(f"Missing request_log for a streaming response: {RawStreamingResponse}!")

    log.debug(
        "OpenAI stream. Applying the following transforms to each chunk: %s", response.transforms
    )
    async for raw_chunk in response.streaming_content:
        chunk = _apply_transforms(raw_chunk, response.transforms)

        chunk_usage = get_token_usage(chunk)
        if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
            token_usage = chunk_usage

        chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)

        try:
            yield f"data: {chunk_str}\n\n"
        except Exception as e:
            yield f"data: {e!s}\n\n"

    end_time = time.monotonic()
    request_log.token_usage = token_usage
    request_log.response_time_ms = int((end_time - start_time) * 1000)
    await request_log.asave()

    yield "data: [DONE]\n\n"


async def _mcp_stream(response: RawStreamingResponse) -> AsyncGenerator[str]:
    """MCP-compliant stream, with post-processing transforms applied to each chunk."""
    log.debug(
        "MCP stream. Applying the following transforms to each chunk: %s", response.transforms
    )
    async for raw_chunk in response.streaming_content:
        chunk = _apply_transforms(raw_chunk, response.transforms)
        chunk_str = chunk.model_dump_json(exclude_none=True)
        yield f"data: {chunk_str}\n\n"


class HttpResponseMiddleware:
    """
    Transform raw responses from the gateway views into valid HTTPResponses.

    This middleware should be applied after all data post-processing
    middleware. Note: expects ``get_response`` to be synchronous (views are
    already wrapped by upstream middleware).
    """

    def __init__(self, get_response: Callable[[ASGIRequest], ViewResult]) -> None:
        self.get_response = get_response

    def __call__(self, request: ASGIRequest) -> JsonResponse | StreamingHttpResponse | HttpResponse:
        """Transform a raw response from a gateway view into a valid HttpResponse.

        If the response is not an instance of ``RawJsonResponse``
        or ``RawStreamingResponse``, it is returned unchanged.
        """
        response = self.get_response(request)

        if isinstance(response, RawJsonResponse):
            # Merge headers from response.headers (may have been modified after init)
            kwargs = response.kwargs.copy()
            kwargs["headers"].update(response.headers)

            if isinstance(response.content, BaseModel):
                response.content = response.content.model_dump(
                    exclude_none=True, exclude_unset=True, mode="json"
                )
            else:
                for k, v in response.content.items():
                    if isinstance(v, BaseModel):
                        # Content can be a dict containing models as values
                        response.content[k] = v.model_dump()
                    elif isinstance(v, (list, tuple)) and any(
                        isinstance(item, BaseModel) for item in v
                    ):
                        # Content can be a dict containing a list of models
                        response.content[k] = [item.model_dump() for item in v]

            return JsonResponse(response.content, **kwargs)

        if isinstance(response, RawStreamingResponse):
            # Merge headers from response.headers (may have been modified after init)
            kwargs = response.kwargs.copy()
            kwargs["headers"].update(response.headers)

            if request.path.startswith("/mcp-servers/"):
                streaming_content = _mcp_stream(response)
            else:
                streaming_content = _openai_stream(response)

            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        return response
