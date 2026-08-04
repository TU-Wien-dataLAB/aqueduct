from collections.abc import Callable

from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse

from gateway.views.utils import RawJsonResponse, RawStreamingResponse, _openai_stream


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
            return JsonResponse(response.content, **response.kwargs)

        if isinstance(response, RawStreamingResponse):
            streaming_content = _openai_stream(
                response.streaming_content, request_log=response.request_log
            )
            return StreamingHttpResponse(streaming_content=streaming_content, **response.kwargs)

        return response
