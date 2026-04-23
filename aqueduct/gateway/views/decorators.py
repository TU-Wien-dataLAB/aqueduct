import base64
import io
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Coroutine, Iterable
from datetime import timedelta
from functools import wraps
from http import HTTPStatus
from typing import Any

import httpx
import litellm
import openai
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.cache import cache
from django.core.files.uploadedfile import UploadedFile
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Sum
from django.http import HttpResponse, StreamingHttpResponse
from django.urls import reverse
from django.utils import timezone
from mcp.types import JSONRPCMessage
from openai.types.chat import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_content_part_param import FileFile
from openai.types.responses import ResponseCreateParams, ToolParam
from pydantic import TypeAdapter, ValidationError
from tos.models import has_user_agreed_latest_tos  # type: ignore[import-untyped]

from gateway.authentication import token_from_request
from gateway.config import (
    MCPServerConfig,
    get_all_model_request_limit_multipliers,
    get_files_api_client,
    get_mcp_config,
    get_router,
    resolve_model_alias,
)
from gateway.views.errors import error_response
from gateway.views.utils import get_response_from_cache, in_wildcard
from management.models import FileObject, Request, Token, VectorStore

log = logging.getLogger("aqueduct")

ViewResult = HttpResponse | StreamingHttpResponse
AsyncView = Callable[..., Coroutine[Any, Any, ViewResult]]
Decorator = Callable[[AsyncView], AsyncView]


def token_authenticated(token_auth_only: bool) -> Decorator:
    def decorator(view_func: AsyncView) -> AsyncView:
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
            unauthorized_response = error_response("Authentication Required", status=401)
            # Authentication Check
            if not (await request.auser()).is_authenticated:
                user = await auth.aauthenticate(request=request)
                if user is not None:
                    request.user = user  # Manually assign user
            else:
                request.user = await request.auser()

            if not getattr(request, "user", None) or not request.user.is_authenticated:
                log.error(
                    "Authentication check failed in ai_gateway_view: "
                    "request.user is not authenticated."
                )
                return unauthorized_response
            log.debug("User %s authenticated.", request.user.email)

            token_key = token_from_request(request)
            if token_auth_only and not token_key:
                log.error("Token not found in request")
                return unauthorized_response

            if token_key:
                token = await sync_to_async(Token.find_by_key)(token_key)
            else:
                # user authenticated but not via token -> use Token.objects for async ORM
                token = (
                    await Token.objects.select_related(
                        "user__profile__org", "service_account__team__org"
                    )
                    .filter(user=request.user)
                    .afirst()
                )

            if not token:
                log.error("Token not found during authentication")
                return unauthorized_response
            kwargs["token"] = token
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


class FileSizeError(Exception):
    """Raised when file size limits are exceeded."""


def _parse_multipart_body(request: ASGIRequest) -> dict[str, Any]:
    """
    Parse the body of a "multipart/form-data" POST request into a Python dict.
    Validate file sizes of the files attached to the request.

    Raises:
        `FileSizeError`: if any file exceeds the `AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB`
          setting, or if the total size of the files exceeds 32 MB.

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


def check_limits(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error("check_limits decorator used without @token_authenticated decorator")
            return error_response("Internal server error", status=500)

        try:
            # Get limits asynchronously
            limits = await sync_to_async(token.get_limit)()
            log.debug("Rate limits for Token %r (ID: %s): %s", token.name, token.id, limits)

            if (
                limits.requests_per_minute is not None
                or limits.input_tokens_per_minute is not None
                or limits.output_tokens_per_minute is not None
            ):
                # Define the time window for usage check (last 60 seconds)
                time_window_start = timezone.now() - timedelta(seconds=60)

                # Build query filter - no model filter (count all requests)
                query_filter: dict[str, Any] = {"token": token, "timestamp__gte": time_window_start}

                # Query recent usage asynchronously using Django's async ORM
                # Get overall token counts
                recent_requests_agg = await Request.objects.filter(**query_filter).aaggregate(
                    request_count=Count("id"),
                    total_input_tokens=Sum("input_tokens"),
                    total_output_tokens=Sum("output_tokens"),
                )

                total_input = recent_requests_agg.get("total_input_tokens", 0) or 0
                total_output = recent_requests_agg.get("total_output_tokens", 0) or 0

                # Get per-model request counts for weighted budget calculation
                model_counts: dict[str, int] = {}
                async for item in (
                    Request.objects.filter(**query_filter)
                    .values("model")
                    .annotate(request_count=Count("id"))
                ):
                    model = item.get("model")
                    if model:
                        model_counts[model] = item["request_count"]

                log.debug(
                    "Recent usage (last 60s) for Token %r: Model counts=%s, Input=%s, Output=%s",
                    token.name,
                    model_counts,
                    total_input,
                    total_output,
                )

                # --- Check Limits ---
                exceeded = []

                # Calculate weighted request count using per-model multipliers
                # "2x Limits" means multiplier=2, so cost = 1/2 = 0.5 per request
                weighted_request_count: float = 0.0
                multipliers = get_all_model_request_limit_multipliers()
                for model, count in model_counts.items():
                    multiplier = multipliers.get(model, 1.0)
                    weighted_request_count += count * (1.0 / multiplier)

                log.debug(
                    "Weighted request count for Token %r: %.2f (base limit: %s)",
                    token.name,
                    weighted_request_count,
                    limits.requests_per_minute,
                )

                if (
                    limits.requests_per_minute is not None
                    and weighted_request_count >= limits.requests_per_minute
                ):
                    exceeded.append(f"Request limit ({limits.requests_per_minute}/min)")

                if (
                    limits.input_tokens_per_minute is not None
                    and total_input >= limits.input_tokens_per_minute
                ):
                    exceeded.append(f"Input token limit ({limits.input_tokens_per_minute}/min)")

                if (
                    limits.output_tokens_per_minute is not None
                    and total_output >= limits.output_tokens_per_minute
                ):
                    exceeded.append(f"Output token limit ({limits.output_tokens_per_minute}/min)")

                if len(exceeded) > 0:
                    error_message = "Rate limit exceeded. " + ", ".join(exceeded) + "."
                    log.warning(
                        "Rate limit exceeded for Token %r (ID: %s). Details: %s",
                        token.name,
                        token.id,
                        error_message,
                    )
                    log.error("Rate limit exceeded - %s", error_message)
                    # Return 429 Too Many Requests
                    return error_response(error_message, status=429)

        except Exception as e:
            log.exception("Error checking rate limits for Token %r: %s", token.name, e)
            return error_response("Internal gateway error checking rate limits", status=500)

        return await view_func(request, *args, **kwargs)

    return wrapper


def log_request(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        is_initialize = kwargs.get("is_initialize", False)

        if request.path.startswith("/mcp-servers/") and not is_initialize:
            kwargs["request_log"] = None
            return await view_func(request, *args, **kwargs)

        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        token = kwargs.get("token")
        request_log = Request(
            token=token,
            model=None if not pydantic_model else pydantic_model.get("model", None),
            # Use the resolved model from self
            timestamp=timezone.now(),
            method=request.method,
            user_agent=request.headers.get("User-Agent", ""),
            ip_address=request.META.get("REMOTE_ADDR"),
            user_id=kwargs.get("user_id", ""),
            # path, Status, time, usage set later in the view or processing steps
        )
        # Calculate and set path (ensure leading slash)
        request_log.path = f"/{request.path.lstrip('/')}"
        kwargs["request_log"] = request_log
        await request_log.asave()
        log.debug("Initial request log object created.")

        start_time = time.monotonic()
        result: HttpResponse | StreamingHttpResponse = await view_func(request, *args, **kwargs)
        end_time = time.monotonic()

        request_log.response_time_ms = int((end_time - start_time) * 1000)
        request_log.status_code = result.status_code

        await request_log.asave()
        return result

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


def check_model_availability(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_model_availability decorator used without @token_authenticated decorator"
            )
            return error_response("Internal server error", status=500)
        body: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not body:
            return await view_func(request, *args, **kwargs)
        model: str | None = body.get("model", None)
        if not model:
            return await view_func(request, *args, **kwargs)
        if await sync_to_async(token.model_excluded)(model):
            log.error("Model not found - %s", model)
            return error_response("Model not found!", status=404)
        return await view_func(request, *args, **kwargs)

    return wrapper


def check_mcp_server_availability(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        if not token:
            log.error(
                "check_mcp_server_availability decorator used without "
                "@token_authenticated decorator"
            )
            return error_response("Internal server error", status=500)
        server_name: str | None = kwargs.get("name")
        if not server_name:
            return await view_func(request, *args, **kwargs)
        if await sync_to_async(token.mcp_server_excluded)(server_name):
            log.error("MCP server not found - %s", server_name)
            return error_response("MCP server not found!", status=404)
        return await view_func(request, *args, **kwargs)

    return wrapper


async def extract_text_with_tika(file_bytes: bytes) -> str:
    """Extract text from file bytes using Tika API."""
    tika_url = f"{getattr(settings, 'TIKA_SERVER_URL', 'http://localhost:9998')}/tika"

    async with httpx.AsyncClient() as client:
        response = await client.put(tika_url, content=file_bytes, headers={}, timeout=30.0)
        response.raise_for_status()
        return response.text


async def file_to_bytes(token: Token | None, file: FileFile) -> bytes:
    """Convert file description to bytes and content type."""
    file_id = file.get("file_id", None)
    file_data = file.get("file_data", None)

    if file_data:
        # file data contains b64 encoded files -> decode as bytes
        try:
            header, file_b64 = file_data.split(
                ",", maxsplit=1
            )  # removes data uri (data:application/pdf;base64,<b64>)
            if not header.startswith("data:"):
                raise ValueError("Incorrect data URI for base64 encoded file.")
            return base64.b64decode(file_b64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 file data: {e}") from e

    elif file_id:
        # file is given as an id of a file object
        try:
            if token and token.service_account:
                file_obj = await FileObject.objects.select_related("token__user").aget(
                    id=file_id, token__service_account__team=token.service_account.team
                )
            elif token:
                file_obj = await FileObject.objects.select_related("token__user").aget(
                    id=file_id, token__user=token.user
                )
            else:
                file_obj = await FileObject.objects.select_related("token__user").aget(id=file_id)

            try:
                client = get_files_api_client()
            except ValueError as e:
                raise ValueError(f"Files API not configured: {e}") from e
            response = await client.files.content(file_obj.id)
            return response.content
        except FileObject.DoesNotExist:
            raise
        except Exception as e:
            raise ValueError(f"Failed to read file with id {file_id}: {e}") from e
        else:
            return response.content
    else:
        raise RuntimeError("Neither 'file_data' nor 'file_id' are given.")


def process_file_content(view_func: AsyncView) -> AsyncView:
    """Decorator to process file content in chat completions using Tika."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        pydantic_model: dict[str, Any] | None = kwargs.get("pydantic_model")
        if not pydantic_model:
            log.error("Invalid request - missing request body")
            return error_response("Invalid request: missing request body", status=400)

        messages = pydantic_model.get("messages", [])
        if not messages:
            return await view_func(request, *args, **kwargs)

        # Process messages to extract text from file content
        total_file_size_bytes = 0
        max_total_size_mb = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB
        max_total_size_bytes = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB * 1024 * 1024
        max_file_mb = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB
        max_file_bytes = settings.AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB * 1024 * 1024
        for message in messages:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue

            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "file":
                    file = FileFile(**content_item.get("file", {}))  # type: ignore[typeddict-item]
                    try:
                        file_bytes = await file_to_bytes(token, file)
                    except FileObject.DoesNotExist:
                        log.exception("File not found")
                        return error_response("File not found", status=404)
                    except Exception as e:
                        # return json response here if there was an error
                        log.exception("Error processing file - %s", e)
                        return error_response(f"Error processing file: {e!s}", status=400)

                    if len(file_bytes) > max_file_bytes:
                        log.error(
                            "File processing error - File too large "
                            "(individual file must be <= %sMB)",
                            max_file_mb,
                        )
                        return error_response(
                            f"Error processing file content: File too large. "
                            f"Individual file must be <= {max_file_mb}MB.",
                            status=400,
                        )
                    total_file_size_bytes += len(file_bytes)
                    if total_file_size_bytes > max_total_size_bytes:
                        log.error(
                            "File processing error - Files too large in total "
                            "(all files must be <= %sMB)",
                            max_total_size_mb,
                        )
                        return error_response(
                            f"Error processing file content: Files too large in total. "
                            f"All files must be <= {max_total_size_mb}MB.",
                            status=400,
                        )

                    # Extract text using Tika
                    try:
                        extracted_text = await extract_text_with_tika(file_bytes)
                    except httpx.HTTPStatusError as e:
                        # return json response here if there was a tika request error
                        log.exception("Tika error extracting text from file - %s", e)
                        return error_response(
                            f"Tika error extracting text from file: {e!s}", status=400
                        )

                    extracted_text = (
                        f"Content of user-uploaded file "
                        f"'{file.get('filename', 'unknown filename')}':"
                        f"\n---\n{extracted_text}\n---"
                    )

                    # Replace file content with extracted text
                    content_item["type"] = "text"
                    content_item["text"] = extracted_text
                    del content_item["file"]

        return await view_func(request, *args, **kwargs)

    return wrapper


def catch_router_exceptions(view_func: AsyncView) -> AsyncView:
    def _r(e: Exception) -> str:
        s = str(e)
        s = re.sub(r"Lite-?[lL][lL][mM]", "Aqueduct", s)  # uppercase
        return re.sub(r"lite-?[lL][lL][mM]", "aqueduct", s)  # lowercase

    def _exception_response(e: Exception, status: int) -> HttpResponse:
        """Convert an openai/litellm exception to an OpenAI-compatible JsonResponse.

        The openai SDK parses ``code``, ``param``, and ``type`` from the
        response body when available (see ``openai.APIError.__init__``).
        LiteLLM exceptions inherit from the corresponding openai classes
        but typically pass ``body=None``, so these will be ``None`` for
        most litellm errors.  We forward whatever is available.
        """
        code = getattr(e, "code", None)
        return error_response(
            message=_r(e),
            error_type=getattr(e, "type", None),
            param=getattr(e, "param", None),
            code=str(code) if code is not None else None,
            status=status,
        )

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # https://docs.litellm.ai/docs/exception_mapping#litellm-exceptions
        # also except equivalent openai exceptions
        try:
            return await view_func(request, *args, **kwargs)
        except (litellm.BadRequestError, openai.BadRequestError) as e:
            log.exception("Bad request - %s", _r(e))
            return _exception_response(e, status=400)
        except (litellm.AuthenticationError, openai.AuthenticationError) as e:
            log.exception("Authentication error - %s", _r(e))
            return _exception_response(e, status=401)
        except (litellm.exceptions.PermissionDeniedError, openai.PermissionDeniedError) as e:
            log.exception("Permission denied - %s", _r(e))
            return _exception_response(e, status=403)
        except (litellm.NotFoundError, openai.NotFoundError) as e:
            log.exception("Not found - %s", _r(e))
            return _exception_response(e, status=404)
        except (litellm.UnprocessableEntityError, openai.UnprocessableEntityError) as e:
            log.exception("Unprocessable entity - %s", _r(e))
            return _exception_response(e, status=422)
        except (litellm.RateLimitError, openai.RateLimitError) as e:
            log.exception("Rate limit exceeded - %s", _r(e))
            return _exception_response(e, status=429)
        except (litellm.Timeout, openai.APITimeoutError) as e:
            log.exception("Timeout - %s", _r(e))
            return _exception_response(e, status=504)
        except (
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            openai.APIConnectionError,
        ) as e:
            log.exception("Service unavailable - %s", _r(e))
            return _exception_response(e, status=503)
        except (litellm.InternalServerError, openai.InternalServerError) as e:
            log.exception("Internal server error - %s", _r(e))
            return _exception_response(e, status=500)
        except (litellm.APIError, openai.APIError) as e:
            # APIError is raised e.g. when user sends extra kwargs in the request body,
            # so we return a 400 Bad request.
            log.exception("API error - %s", _r(e))
            return _exception_response(e, status=400)
        except Exception as e:
            log.exception("Unexpected error - %s", _r(e))
            return error_response(_r(e), error_type="server_error", status=502)

    return wrapper


def tos_accepted(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        if settings.TOS_ENABLED and settings.TOS_GATEWAY_VALIDATION:
            token: Token | None = kwargs.get("token")
            if not token:
                log.error("tos_accepted decorator used without @token_authenticated decorator")
                return error_response("Internal server error", status=500)
            key_version = cache.get("django:tos:key_version")
            user_id = token.user.id

            skip: bool = cache.get(
                f"django:tos:skip_tos_check:{user_id}", False, version=key_version
            )

            if not skip:
                user_agreed = cache.get(f"django:tos:agreed:{user_id}", None, version=key_version)
                if user_agreed is None:
                    user_agreed = await sync_to_async(has_user_agreed_latest_tos)(request.user)

                if not user_agreed:
                    log.error("Terms of service agreement required")
                    return error_response(
                        "In order to use the API you have to agree to the terms of service!",
                        status=403,
                    )

        return await view_func(request, *args, **kwargs)

    return wrapper


def mcp_transport_security(view_func: AsyncView) -> AsyncView:
    """Validate MCP transport security (DNS rebinding protection).

    Validates:
    - Host header (DNS rebinding protection)
    - Origin header (CSRF protection)
    - Content-Type header for POST requests

    Returns appropriate status codes:
    - 421: Invalid Host header
    - 403: Invalid Origin header
    - 400: Invalid Content-Type header
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # Skip validation if DNS rebinding protection is disabled
        if not getattr(settings, "MCP_ENABLE_DNS_REBINDING_PROTECTION", True):
            return await view_func(request, *args, **kwargs)

        log.debug("MCP request headers: %s", dict(request.headers))

        # Validate Content-Type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            log.debug("POST request Content-Type: %r", content_type)
            if not content_type.lower().startswith("application/json"):
                log.error("Invalid Content-Type header: %s", content_type)
                return error_response("Invalid Content-Type header", status=400)

        # Validate Host header against allowed values
        allowed_hosts = getattr(settings, "MCP_ALLOWED_HOSTS", [])
        host = request.headers.get("host")

        if not host:
            log.error("Missing Host header in request")
            return error_response("Invalid Host header", status=421)

        host_valid = in_wildcard(host, allowed_hosts)
        if not host_valid:
            log.error("Invalid Host header: %s", host)
            return error_response("Invalid Host header", status=421)

        # Validate Origin header against allowed values
        # Origin can be absent for same-origin requests, so it\'s only validated if present
        allowed_origins = getattr(settings, "MCP_ALLOWED_ORIGINS", [])
        origin = request.headers.get("origin")
        if origin:
            origin_valid = in_wildcard(origin, allowed_origins)
            if not origin_valid:
                log.error("Invalid Origin header: %s", origin)
                return error_response("Invalid Origin header", status=403)

        return await view_func(request, *args, **kwargs)

    return wrapper


def parse_jsonrpc_message(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        session_id = request.headers.get("Mcp-Session-Id")
        kwargs["session_id"] = session_id

        if request.method != "POST":
            if not session_id:
                log.error("Session ID required for MCP server %r", kwargs.get("name"))
                return error_response("Mcp-Session-Id header required", status=400)

            return await view_func(request, *args, request_log=None, **kwargs)

        data = kwargs["pydantic_model"]
        # For mcp requests, timeout should not be passed to the JSON RPC Message
        data.pop("timeout", None)
        json_rpc_message = JSONRPCMessage.model_validate(data)
        is_initialize = (
            hasattr(json_rpc_message.root, "method")
            and json_rpc_message.root.method == "initialize"
        )

        if not is_initialize and not session_id:
            log.error("Session ID required for MCP server %r", kwargs.get("name"))
            return error_response("Mcp-Session-Id header required", status=400)

        kwargs["json_rpc_message"] = json_rpc_message
        kwargs["is_initialize"] = is_initialize

        return await view_func(request, *args, **kwargs)

    return wrapper


def validate_response_id(view_func: AsyncView) -> AsyncView:
    @wraps(view_func)
    async def wrapper(
        request: ASGIRequest, response_id: str, *args: Any, **kwargs: Any
    ) -> ViewResult:
        token = kwargs.get("token")
        if not token:
            log.error("validate_response_id decorator used without @token_authenticated decorator")
            return error_response("Internal server error", status=500)

        response = get_response_from_cache(response_id)
        if not response:
            return error_response("Response not found", status=404)

        if response["email"] != token.user.email:
            return error_response("Response not found", status=404)

        return await view_func(request, response_id, *args, **kwargs)

    return wrapper


async def _validate_mcp_tool(
    request: ASGIRequest, token: Token, tool: ToolParam
) -> ViewResult | None:
    # Note: mypy doesn't recognise the types of `ToolParam` attributes correctly
    server_name: str = tool.get("server_label")  # type: ignore[assignment]
    if await sync_to_async(token.mcp_server_excluded)(server_name):
        log.error("MCP server not found - %s", server_name)
        return error_response(f"MCP server not found - {server_name}", status=404)

    server_url = tool.get("server_url")
    if not server_url:
        if not settings.RESPONSES_API_ALLOW_EXTERNAL_MCP_SERVERS:
            log.exception("MCP server not found - %s", server_name)
            return error_response(f"MCP server not found - {server_name}", status=404)
        return None

    mcp_config = get_mcp_config()
    server_config: MCPServerConfig | None = None
    for config_name, config in mcp_config.items():
        expected = request.build_absolute_uri(
            reverse("gateway:mcp_server", kwargs={"name": config_name})
        )
        if server_url == expected:
            server_config = config
            break

    if not server_config:
        if not settings.RESPONSES_API_ALLOW_EXTERNAL_MCP_SERVERS:
            log.exception("MCP server not found - %s", server_name)
            return error_response(f"MCP server not found - {server_name}", status=404)
        return None

    tool["server_url"] = server_config["url"]  # type: ignore[typeddict-unknown-key]
    return None


async def _validate_file_search_tool(token: Token, tool: ToolParam) -> ViewResult | None:
    # Note: mypy doesn't recognise the types of `ToolParam` attributes correctly
    vector_store_ids: list[str] = tool.get("vector_store_ids", [])  # type: ignore[assignment]
    if not vector_store_ids:
        return None

    # Deduplicate to avoid false negatives in count check
    unique_vs_ids = list(set(vector_store_ids))
    # Verify ownership - users can only use their own vector stores
    if token.service_account:
        vs_count = await VectorStore.objects.filter(
            id__in=unique_vs_ids, token__service_account__team=token.service_account.team
        ).acount()
    else:
        vs_count = await VectorStore.objects.filter(
            id__in=unique_vs_ids, token__user=token.user
        ).acount()

    if vs_count != len(unique_vs_ids):
        return error_response("One or more vector stores not found", status=404)
    return None


def _validate_native_tool(tool_type: str | None) -> ViewResult | None:
    if tool_type not in settings.RESPONSES_API_ALLOWED_NATIVE_TOOLS:
        return error_response(f"Invalid tool type: {tool_type}", status=400)
    return None


async def _validate_tool(request: ASGIRequest, token: Token, tool: ToolParam) -> ViewResult | None:
    tool_type = tool.get("type")

    if tool_type in {"function", "custom"}:
        return None

    if tool_type == "mcp":
        return await _validate_mcp_tool(request, token, tool)

    if tool_type == "file_search":
        return await _validate_file_search_tool(token, tool)

    return _validate_native_tool(tool_type)


def check_tool_availability(view_func: AsyncView) -> AsyncView:
    """
    Validate tool availability and configuration for Responses API requests.

    Checks that MCP server tools are accessible to the user's token and properly
    configured. Validates server URLs for Aqueduct-managed MCP servers and ensures
    native tools are allowed in settings. It prevents unauthorized access
    to MCP servers and ensures tools are correctly configured.

    Used on Responses API endpoints that accept tools in the request body.
    Requires @token_authenticated and @parse_body decorators.
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        token: Token | None = kwargs.get("token")
        pydantic_model: ResponseCreateParams | None = kwargs.get("pydantic_model")
        if not token or not pydantic_model:
            return error_response("Invalid request", status=400)

        tools: Iterable[ToolParam] = pydantic_model.get("tools") or []
        for tool in tools:
            error = await _validate_tool(request, token, tool)
            if error:
                return error

        return await view_func(request, *args, **kwargs)

    return wrapper


def require_files_api_client(view_func: AsyncView) -> AsyncView:
    """Decorator that injects a files API client into the view kwargs, or returns 503.

    Uses late-bound import of get_files_api_client so that tests can
    patch it on the calling module (e.g. gateway.views.vector_stores).
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        # Look up get_files_api_client from the module where view_func is defined,
        # so tests patching that module's reference will be respected.
        view_module = sys.modules.get(view_func.__module__)
        _get_client = getattr(view_module, "get_files_api_client", get_files_api_client)

        try:
            client = _get_client()
        except ValueError:
            return error_response("Vector Store API not configured", status=503)
        kwargs["client"] = client
        return await view_func(request, *args, **kwargs)

    return wrapper


def _lookup_relay_model_name(requested_model: str) -> str | None:
    """Return upstream relay model for a configured deployment, else None."""
    router = get_router()
    if not router:
        return requested_model

    requested_model = resolve_model_alias(requested_model)
    deployment: litellm.Deployment | None = router.get_deployment(model_id=requested_model)
    if not deployment:
        return None

    litellm_params = getattr(deployment, "litellm_params", None)
    deployment_model = getattr(litellm_params, "model", None)
    if not deployment_model:
        return None

    relay_model: str
    relay_model, _, _, _ = litellm.get_llm_provider(deployment_model)
    return relay_model


def get_relay_model_name(requested_model: str) -> str:
    """
    Map a requested model name to the actual upstream model name.

    Uses the LiteLLM router configuration to find the deployment.
    """
    relay_model = _lookup_relay_model_name(requested_model)
    if relay_model is None:
        return requested_model
    return relay_model


def rewrite_batch_file_models(content: bytes) -> bytes:
    """
    Rewrite model names in a batch input JSONL file.

    Each line is a JSON object with structure:
    {"custom_id": "...", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "relay-model-name", ...}}

    This function:
    1. Ensures custom_id is a string type (required by OpenAI Batch API)
    2. Replaces the model name in each request body with the actual upstream model name
    3. Rejects unknown models with a ValueError that includes custom_id
    """
    lines = content.decode("utf-8").splitlines()
    rewritten_lines = []

    for line in lines:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            body = request.get("body", {})
            if "model" in body:
                original_model = body["model"]
                relay_model = _lookup_relay_model_name(original_model)
                if relay_model is None:
                    custom_id = request.get("custom_id", "<missing custom_id>")
                    raise ValueError(
                        f"Unknown model '{original_model}' for custom_id '{custom_id}'"
                    )
                body["model"] = relay_model
                request["body"] = body
            if "custom_id" in request:
                request["custom_id"] = str(request["custom_id"])
            rewritten_lines.append(json.dumps(request, separators=(",", ":")))
        except json.JSONDecodeError:
            # Keep invalid lines as-is (will fail validation anyway)
            rewritten_lines.append(line)

    return ("\n".join(rewritten_lines) + "\n").encode("utf-8")


def extract_preview(content: bytes, num_lines: int = 10) -> str:
    """Extract first 10 lines from file content for preview."""
    try:
        lines = content.decode("utf-8").splitlines()[:num_lines]
        return "\n".join(lines)
    except UnicodeDecodeError:
        return "[Binary content - no preview available]"


def process_batch_file(view_func: AsyncView) -> AsyncView:
    """
    Decorator for batch file upload that processes file content before proxying.

    This decorator (applied after @parse_body):
    1. Reads the file content ONCE from the uploaded file
    2. Extracts a preview (first 10 lines) of the original content
    3. For batch files: rewrites model names using router config
    4. Passes processed data to the view via kwargs:
       - file_content: bytes - the raw file content (or rewritten for batch)
       - file_preview: str | None - first 10 lines for batch files

    This follows the existing decorator pattern (like @parse_body) of reading
    data once and passing it via kwargs to avoid multiple reads.
    """

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args: Any, **kwargs: Any) -> ViewResult:
        if request.method != "POST":
            return await view_func(request, *args, **kwargs)

        pydantic_model = kwargs.get("pydantic_model")
        if not pydantic_model:
            return await view_func(request, *args, **kwargs)

        uploaded = pydantic_model.get("file")
        purpose = pydantic_model.get("purpose")

        if not uploaded:
            return await view_func(request, *args, **kwargs)

        # Read file content ONCE
        content = uploaded.read()

        if purpose == "batch":
            kwargs["file_preview"] = extract_preview(content)
            try:
                kwargs["file_content"] = rewrite_batch_file_models(content)
            except ValueError as e:
                return error_response(f"Batch file validation failed: {e!s}", status=400)
        else:
            kwargs["file_content"] = content
            kwargs["file_preview"] = ""

        return await view_func(request, *args, **kwargs)

    return wrapper
