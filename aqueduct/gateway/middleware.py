import time
from collections.abc import AsyncGenerator, Callable
from functools import reduce

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse

from gateway.views.utils import RawJsonResponse, RawStreamingResponse, get_token_usage
from management.models import Usage


async def _openai_stream(response: RawStreamingResponse) -> AsyncGenerator[str, None]:
    """Process streaming response with transforms and log token usage."""
    token_usage = Usage(0, 0)
    start_time = time.monotonic()

    # TODO: we don't log mcp requests! -> handle this case separately

    async for raw_chunk in response.streaming_content:
        # Apply all registered transforms
        chunk = reduce(lambda obj, tr: tr(obj), response.transforms, raw_chunk)

        # Retrieve token usage
        chunk_usage = get_token_usage(chunk)
        if chunk_usage.input_tokens > 0 or chunk_usage.output_tokens > 0:
            token_usage = chunk_usage

        # Format as SSE (suitable for StreamingHttpResponse)
        chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
        try:
            yield f"data: {chunk_str}\n\n"
        except Exception as e:
            yield f"data: {e!s}\n\n"

    end_time = time.monotonic()
    request_log = response.request_log
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

            streaming_content = _openai_stream(response)
            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        return response
