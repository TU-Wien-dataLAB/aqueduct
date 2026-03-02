from django.http import JsonResponse
from openai.types import ErrorObject


def error_response(
    message: str,
    error_type: str | None = None,
    param: str | None = None,
    code: str | None = None,
    status: int = 400,
) -> JsonResponse:
    """Return an OpenAI-compatible error response."""
    if error_type is None:
        error_type = _status_to_error_type(status)
    error = ErrorObject(message=message, type=error_type, param=param, code=code)
    return JsonResponse({"error": error.model_dump(exclude_none=True)}, status=status)


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
