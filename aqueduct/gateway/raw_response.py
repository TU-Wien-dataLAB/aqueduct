import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from django.conf import settings
from django.core.cache import caches
from django.http.response import ResponseHeaders
from openai.types import ErrorObject
from pydantic import BaseModel

from management.models import Request

log = logging.getLogger("aqueduct")


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


def in_wildcard(value: str | None, allowed_values: list[str]) -> bool:
    """Check if a value is in a list of allowed values or matches a wildcard pattern."""
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


def register_response_in_cache(response_id: str | None, model: str, email: str) -> None:
    """Registers a response in the cache for later retrieval."""
    if not response_id:
        log.warning("Missing response data: id=%s, model=%s", response_id, model)
        raise ValueError("Missing response_id")

    cache_key = f"response:{response_id}"
    cache_value = {"model": model, "email": email}

    response_cache = caches["default"]
    response_cache.set(cache_key, cache_value, timeout=settings.RESPONSES_API_TTL_SECONDS)
    log.debug("Registered response %s for user %s with model %s", response_id, email, model)


def get_response_from_cache(response_id: str) -> dict[str, Any] | None:
    """Retrieves a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    result: dict[str, Any] | None = response_cache.get(cache_key)
    return result


def delete_response_from_cache(response_id: str) -> None:
    """Deletes a response from the cache."""
    cache_key = f"response:{response_id}"
    response_cache = caches["default"]
    response_cache.delete(cache_key)
