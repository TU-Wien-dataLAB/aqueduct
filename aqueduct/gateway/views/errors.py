from django.http import JsonResponse


def error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    status: int = 400,
) -> JsonResponse:
    """Return an OpenAI-compatible error response."""
    error = {"message": message, "type": error_type}
    if param:
        error["param"] = param
    return JsonResponse({"error": error}, status=status)
