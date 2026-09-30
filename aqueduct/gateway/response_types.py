from collections.abc import AsyncIterator, Callable
from typing import Any

from django.http.response import ResponseHeaders
from openai.types import ErrorObject
from pydantic import BaseModel

from management.models import Request


class RawJsonResponse:
    """A wrapper for data that can be turned into a JSONResponse."""

    def __init__(self, data: dict[str, Any] | BaseModel, **kwargs: Any) -> None:
        if not isinstance(data, (dict, BaseModel)):
            raise TypeError("RawJsonResponse data has to be a dict or a pydantic BaseModel")

        self.content = data
        self.kwargs = kwargs or {}
        self.content_type = self.kwargs.setdefault("content_type", "application/json")
        # Just to be on the safe side, make header keys case-insensitive:
        self.headers = ResponseHeaders(self.kwargs.setdefault("headers", {}))
        # The following mimics the BaseHttpResponse behaviour (argument called "status"
        # is assigned to the "status_code" attribute)
        self.status_code = self.kwargs.get("status", 200)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} status_code={self.status_code}>"


class RawStreamingResponse:
    """A wrapper for streaming data that can be turned into a StreamingHttpResponse."""

    def __init__(
        self,
        streaming_content: AsyncIterator[Any],
        request_log: Request | None,
        transforms: list[Callable[[Any], Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(streaming_content, AsyncIterator):
            raise TypeError("RawStreamResponse streaming_content has to be async iterable")

        self.streaming_content = streaming_content
        self.request_log = request_log
        self.transforms = transforms or []
        self.kwargs = kwargs or {}
        self.content_type = self.kwargs.setdefault("content_type", "text/event-stream")
        # Just to be on the safe side, make header keys case-insensitive:
        self.headers = ResponseHeaders(self.kwargs.setdefault("headers", {}))
        # The following mimics the BaseHttpResponse behaviour (argument called "status"
        # is assigned to the "status_code" attribute)
        self.status_code = self.kwargs.get("status", 200)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} status_code={self.status_code}>"


def error_response(
    message: str,
    error_type: str | None = None,
    param: str | None = None,
    code: str | None = None,
    status: int = 400,
) -> RawJsonResponse:
    """Return an OpenAI-compatible error response."""
    if error_type is None:
        error_type = _status_to_error_type(status)
    error = ErrorObject(message=message, type=error_type, param=param, code=code)
    return RawJsonResponse({"error": error.model_dump(exclude_none=True)}, status=status)


def _status_to_error_type(status: int) -> str:
    """Map HTTP status codes to OpenAI error types."""
    status_map = {
        400: "invalid_request_error",
        401: "invalid_request_error",
        403: "permission_denied_error",
        404: "not_found_error",
        410: "invalid_request_error",
        421: "invalid_request_error",
        422: "invalid_request_error",
        429: "rate_limit_error",
        500: "server_error",
        503: "server_error",
        504: "timeout_error",
    }
    return status_map.get(status, "invalid_request_error")
