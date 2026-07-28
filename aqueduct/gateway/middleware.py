from collections.abc import Callable
from typing import Any, reveal_type

from asgiref.sync import iscoroutinefunction
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.utils.decorators import sync_only_middleware

from gateway.views.utils import RawJsonResponse, RawStreamingResponse, _openai_stream


@sync_only_middleware
def http_response_middleware(
    get_response: Callable[[ASGIRequest], JsonResponse | StreamingHttpResponse],
) -> Callable[[ASGIRequest], Any]:
    """
    Transform raw responses from the gateway views into valid HTTPResponses.

    This middleware should be applied after all data post-processing
    middleware.
    """
    if iscoroutinefunction(get_response):
        # async def middleware(request: ASGIRequest) -> JsonResponse | StreamingHttpResponse:
        #     response = await get_response(request)
        #
        #     if isinstance(response, RawJsonResponse):
        #         return JsonResponse(response.data, **response.kwargs)
        #
        #     if isinstance(response, RawStreamingResponse):
        #         streaming_content = _openai_stream(response.stream, request_log=response.request_log)
        #         return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)
        #
        #
        #     print(f"\n\nASYNC response: {reveal_type(response)}\n\n")
        #     return response
        raise TypeError("Got a coroutine, expected a sync function")

    def middleware(request: ASGIRequest) -> JsonResponse | StreamingHttpResponse:
        response = get_response(request)

        if isinstance(response, RawJsonResponse):
            return JsonResponse(response.data, **response.kwargs)

        if isinstance(response, RawStreamingResponse):
            streaming_content = _openai_stream(response.stream, request_log=response.request_log)
            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        print(f"\n\nSYNC response: {reveal_type(response)}\n\n")
        return response

    return middleware


class HTTPResponseMiddleware:
    """
    Transform raw responses from the gateway views into valid HTTPResponses.

    This middleware should be applied after all data post-processing
    middleware.
    """

    # sync_capable = False
    # async_capable = True

    def __init__(
        self, get_response: Callable[[ASGIRequest], JsonResponse | StreamingHttpResponse]
    ) -> None:
        self.get_response = get_response
        # if iscoroutinefunction(get_response):
        #     markcoroutinefunction(self)

    def __call__(self, request: ASGIRequest) -> JsonResponse | StreamingHttpResponse:
        """Transform a raw response from a gateway view into a valid HTTPResponse.

        If the response is not an instance of ``RawJsonResponse``
        or ``RawStreamingResponse``, it is returned unchanged.
        """
        response = self.get_response(request)

        if isinstance(response, RawJsonResponse):
            return JsonResponse(response.data, **response.kwargs)

        if isinstance(response, RawStreamingResponse):
            streaming_content = _openai_stream(response.stream, request_log=response.request_log)
            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        return response
