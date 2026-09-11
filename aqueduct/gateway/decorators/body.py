import io
import json
import logging
from functools import wraps
from http import HTTPStatus
from typing import Any

from django.conf import settings
from django.core.files.uploadedfile import UploadedFile
from django.core.handlers.asgi import ASGIRequest
from openai.types.chat import ChatCompletionStreamOptionsParam
from pydantic import TypeAdapter, ValidationError

from gateway.config import resolve_model_alias
from gateway.decorators.types import AsyncView, Decorator, ViewResult
from gateway.raw_response import error_response

log = logging.getLogger("aqueduct")


class FileSizeError(Exception):
    """Raised when file size limits are exceeded."""


def _parse_multipart_body(request: ASGIRequest) -> dict[str, Any]:
    """
    Parse the body of a "multipart/form-data" POST request into a Python dict.
    Validate file sizes of the files attached to the request.

    Raises:
        `FileSizeError`: if any file exceeds the `AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB`
          setting, or if the total size of the files exceeds `AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB`.

    Returns:
        dict with request's body items and files.
    """
    data = {}
    for key, value in request.POST.items():
        try:
            data[key] = json.loads(value)  # type: ignore[arg-type]
        except (TypeError, json.JSONDecodeError):
            if key == "timestamp_granularities[]":
                # OpenAI SDK turns timestamp_granularities into timestamp_granularities[]
                # when sending HTTP request
                # This has to be undone to avoid errors with subsequent
                # client.audio.transcriptions.create call
                data["timestamp_granularities"] = request.POST.getlist(key)
            else:
                data[key] = value

    max_file_size_mb = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB
    max_file_bytes = int(settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB * 1024 * 1024)

    max_total_size_mb = settings.AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB
    max_total_size_bytes = max_total_size_mb * 1024 * 1024
    total_file_size_bytes = 0

    for key, file in request.FILES.items():
        assert isinstance(file, UploadedFile) and isinstance(file.size, int)
        if file.size > max_file_bytes:
            log.error("File in request too large")
            raise FileSizeError(f"File '{key}' exceeds maximum size of {max_file_size_mb}MB")

        total_file_size_bytes += file.size
        if total_file_size_bytes > max_total_size_bytes:
            log.error("Files in request too large")
            raise FileSizeError(f"Total file size exceeds maximum of {max_total_size_mb}MB")

        data[key] = file.read()

    return data


def parse_body(model: TypeAdapter[Any]) -> Decorator:
    """
    Decorator that parses and validates HTTP request bodies for async view functions.

    Only attempts to parse the body of POST requests, otherwise does nothing.
    Handles requests with "application/json" and "multipart/form-data" content types.
    The "pydantic_model" dict with parsed and validated data is passed in the kwargs
    to the view function.
    Additionally, the timeout for the router is added to the parsed data dict.
    If the body contains the field "user_id", it is removed from the parsed data
    and added to kwargs instead.

    Args:
        model: The pydantic model used for request body validation.
    Returns:
        Decorator function that wraps async view functions.
    """

    def decorator(view_func: AsyncView) -> AsyncView:
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
            if request.method != "POST":
                return await view_func(request, *args, **kwargs)

            if request.body is None:
                log.error("Request body is None")
                return error_response("Missing request body", status=400)

            content_type = request.headers.get("content-type", "")

            if content_type.startswith("application/json"):
                body = request.body.decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    log.exception("JSON decode error: %s, body was: %r", e, request.body)
                    return error_response(f"Invalid JSON: {e!s}", status=400)
            elif content_type.startswith("multipart/form-data"):
                try:
                    data = _parse_multipart_body(request)
                except FileSizeError as e:
                    return error_response(str(e), status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            else:
                log.error("Unsupported Content-Type: %s", content_type)
                return error_response(
                    f"Unsupported Content-Type: {content_type}",
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )

            # "user_id" can be sent in the body (it is saved in the request log),
            # but we do not want to leave it in pydantic_model.
            kwargs["user_id"] = data.pop("user_id", "")

            try:
                model.validate_python(data)
            except ValidationError as e:
                log.exception("Validation error: %s", e)
                error_messages = ", ".join(
                    f"{err['loc'][0] if err['loc'] else 'field'}: {err['msg']}"
                    for err in e.errors()
                )
                return error_response(error_messages, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                log.exception("Request body parse error: %s, data was: %r", e, data)
                return error_response(f"Failed to parse the request body: {e!s}", status=400)

            # If there are files sent with the request (i.e. content type is "multipart/form-data"),
            # update bytes to BytesIO because pydantic TypeAdapter has problems with BytesIO.
            # OpenAI usually expects a name for the file object (not just bytes).
            # This only works if the field is also typed as bytes.
            # Additionally, add the size information - for convenience.
            for key, file in request.FILES.items():
                assert isinstance(file, UploadedFile) and isinstance(file.size, int)
                buffer: io.BytesIO = io.BytesIO(data[key])
                buffer.name = file.name
                buffer.size = file.size  # type: ignore[attr-defined]
                data[key] = buffer

            kwargs["pydantic_model"] = data
            kwargs["pydantic_model"]["timeout"] = settings.RELAY_REQUEST_TIMEOUT
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


def ensure_usage(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        model: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not model:
            return await view_func(request, *args, **kwargs)
        stream = model.get("stream", False)
        if stream:
            stream_options = model.get("stream_options", None)
            if not stream_options:
                model["stream_options"] = ChatCompletionStreamOptionsParam(include_usage=True)
            else:
                model["stream_options"]["include_usage"] = True
        return await view_func(request, *args, **kwargs)

    return wrapper


def resolve_alias(view_func: AsyncView) -> AsyncView:
    """Resolve model aliases to actual model names before processing."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not pydantic_model:
            return await view_func(request, *args, **kwargs)

        model_or_alias: str | None = pydantic_model.get("model", None)
        if model_or_alias:
            resolved_model = resolve_model_alias(model_or_alias)
            pydantic_model["model"] = resolved_model
            log.debug("Resolved model %r to %r", model_or_alias, resolved_model)

        return await view_func(request, *args, **kwargs)

    return wrapper
