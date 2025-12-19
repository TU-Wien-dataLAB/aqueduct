import base64
import io
import json
import logging
import re
import time
from datetime import timedelta
from functools import wraps
from http import HTTPStatus

import httpx
import litellm
import openai
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Sum
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.utils import timezone
from mcp.types import JSONRPCMessage
from openai.types.chat import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_content_part_param import FileFile
from pydantic import TypeAdapter, ValidationError
from tos.middleware import cache
from tos.models import has_user_agreed_latest_tos

from gateway.authentication import token_from_request
from gateway.config import resolve_model_alias
from gateway.views.utils import in_wildcard
from management.models import FileObject, Request, Token

log = logging.getLogger("aqueduct")


def token_authenticated(token_auth_only: bool):
    def decorator(view_func):
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args, **kwargs):
            unauthorized_response = JsonResponse({"error": "Authentication Required"}, status=401)
            # Authentication Check
            if not (await request.auser()).is_authenticated:
                user = await auth.aauthenticate(request=request)
                if user is not None:
                    request.user = user  # Manually assign user
            else:
                request.user = await request.auser()

            if not getattr(request, "user", None) or not request.user.is_authenticated:
                log.error(
                    "Authentication check failed in ai_gateway_view: request.user "
                    "is not authenticated."
                )
                return unauthorized_response
            log.debug(f"User {request.user.email} authenticated.")

            token_key = token_from_request(request)
            if token_auth_only and not token_key:
                log.error("Token not found in request")
                return unauthorized_response

            if token_key:
                token = await sync_to_async(Token.find_by_key)(token_key)
            else:
                # user is authenticated but not via token -> take first token via Token.objects to use async ORM
                token = await Token.objects.filter(user=request.user).afirst()

            if not token:
                log.error("Token not found during authentication")
                return unauthorized_response
            kwargs["token"] = token
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


class FileSizeError(Exception):
    """Raised when file size limits are exceeded."""

    pass


def _parse_multipart_body(request: ASGIRequest) -> dict:
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
            data[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            if key == "timestamp_granularities[]":
                # OpenAI SKD turns timestamp_granularities into timestamp_granularities[] when sending HTTP request
                # This has to be undone to avoid errors with the subsequent client.audio.transcriptions.create call
                data["timestamp_granularities"] = request.POST.getlist(key)
            else:
                data[key] = value

    max_file_size_mb = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB
    max_file_bytes = int(settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB * 1024 * 1024)

    max_total_size_mb = settings.AQUEDUCT_FILES_API_MAX_TOTAL_SIZE_MB
    max_total_size_bytes = max_total_size_mb * 1024 * 1024
    total_file_size_bytes = 0

    for key, file in request.FILES.items():
        if file.size > max_file_bytes:
            log.error("File in request too large")
            raise FileSizeError(f"File '{key}' exceeds maximum size of {max_file_size_mb}MB")

        total_file_size_bytes += file.size
        if total_file_size_bytes > max_total_size_bytes:
            log.error("Files in request too large")
            raise FileSizeError(f"Total file size exceeds maximum of {max_total_size_mb}MB")

        data[key] = file.read()

    return data


def parse_body(model: TypeAdapter):
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

    def decorator(view_func):
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args, **kwargs):
            if request.method != "POST":
                return await view_func(request, *args, **kwargs)

            if request.body is None:
                log.error("Request body is None")
                return JsonResponse({"error": "Missing request body"}, status=400)

            content_type = request.headers.get("content-type", "")

            if content_type.startswith("application/json"):
                body = request.body.decode("utf-8")
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as e:
                    log.error(f"JSON decode error: {str(e)}, body was: {request.body!r}")
                    return JsonResponse({"error": f"Invalid JSON: {str(e)}"}, status=400)
            elif content_type.startswith("multipart/form-data"):
                try:
                    data = _parse_multipart_body(request)
                except FileSizeError as e:
                    return JsonResponse(
                        {"error": str(e)}, status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                    )
            else:
                log.error(f"Unsupported Content-Type: {content_type}")
                return JsonResponse(
                    {"error": f"Unsupported Content-Type: {content_type}"},
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                )

            # "user_id" can be sent in the body (it is saved in the request log),
            # but we do not want to leave it in pydantic_model.
            kwargs["user_id"] = data.pop("user_id", "")

            try:
                model.validate_python(data)
            except ValidationError as e:
                log.error(f"Validation error: {e}")
                return JsonResponse({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            except Exception as e:
                log.error(f"Request body parse error: {str(e)}, data was: {data!r}")
                return JsonResponse(
                    {"error": f"Failed to parse the request body: {str(e)}"}, status=400
                )

            # If there are files sent with the request (i.e. content type is "multipart/form-data"),
            # update bytes to BytesIO because pydantic TypeAdapter has problems with BytesIO.
            # OpenAI usually expects a name for the file object (not just bytes).
            # This only works if the field is also typed as bytes.
            # Additionally, add the size information - for convenience.
            for key, file in request.FILES.items():
                buffer: io.BytesIO = io.BytesIO(data[key])
                buffer.name = file.name
                buffer.size = file.size
                data[key] = buffer

            kwargs["pydantic_model"] = data
            kwargs["pydantic_model"]["timeout"] = settings.RELAY_REQUEST_TIMEOUT
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


def ensure_usage(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        model: dict | None = kwargs.get("pydantic_model", None)
        if not model:
            return await view_func(request, *args, **kwargs)
        else:
            stream = model.get("stream", False)
            if stream:
                stream_options = model.get("stream_options", None)
                if not stream_options:
                    model["stream_options"] = ChatCompletionStreamOptionsParam(include_usage=True)
                else:
                    model["stream_options"]["include_usage"] = True
            return await view_func(request, *args, **kwargs)

    return wrapper


def check_limits(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        token: Token | None = kwargs.get("token", None)
        if not token:
            log.error("Token not found")
            return JsonResponse({"error": "Token not found"}, status=404)

        try:
            # Get limits asynchronously
            limits = await sync_to_async(token.get_limit)()
            log.debug(f"Rate limits for Token '{token.name}' (ID: {token.id}): {limits}")

            if (
                limits.requests_per_minute is not None
                or limits.input_tokens_per_minute is not None
                or limits.output_tokens_per_minute is not None
            ):
                # Define the time window for usage check (last 60 seconds)
                time_window_start = timezone.now() - timedelta(seconds=60)

                # Query recent usage asynchronously using Django's async ORM
                recent_requests_agg = await Request.objects.filter(
                    token=token, timestamp__gte=time_window_start
                ).aaggregate(
                    request_count=Count("id"),
                    total_input_tokens=Sum("input_tokens"),
                    total_output_tokens=Sum("output_tokens"),
                )

                request_count = recent_requests_agg.get("request_count", 0) or 0
                total_input = recent_requests_agg.get("total_input_tokens", 0) or 0
                total_output = recent_requests_agg.get("total_output_tokens", 0) or 0

                log.debug(
                    f"Recent usage (last 60s) for Token '{token.name}': "
                    f"Requests={request_count}, Input={total_input}, Output={total_output}"
                )

                # --- Check Limits ---
                exceeded = []

                if (
                    limits.requests_per_minute is not None
                    and request_count >= limits.requests_per_minute
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
                        f"Rate limit exceeded for Token '{token.name}' (ID: {token.id}). "
                        f"Details: {error_message}"
                    )
                    log.error(f"Rate limit exceeded - {error_message}")
                    # Return 429 Too Many Requests
                    return JsonResponse({"error": error_message}, status=429)

        except Exception as e:
            log.error(f"Error checking rate limits for Token '{token.name}': {e}", exc_info=True)
            log.error("Internal gateway error checking rate limits")
            return JsonResponse(
                {"error": "Internal gateway error checking rate limits"}, status=500
            )

        return await view_func(request, *args, **kwargs)

    return wrapper


def log_request(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        is_initialize = kwargs.get("is_initialize", False)

        if request.path.startswith("/mcp-servers/") and not is_initialize:
            kwargs["request_log"] = None
            return await view_func(request, *args, **kwargs)

        pydantic_model: dict | None = kwargs.get("pydantic_model", None)
        token = kwargs.get("token", None)
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


def resolve_alias(view_func):
    """Resolve model aliases to actual model names before processing."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        pydantic_model: dict | None = kwargs.get("pydantic_model", None)
        if not pydantic_model:
            return await view_func(request, *args, **kwargs)

        model_or_alias: str | None = pydantic_model.get("model", None)
        if model_or_alias:
            # Resolve alias to actual model name
            resolved_model = resolve_model_alias(model_or_alias)
            pydantic_model["model"] = resolved_model
            log.debug(f"Resolved model '{model_or_alias}' to '{resolved_model}'")

        return await view_func(request, *args, **kwargs)

    return wrapper


def check_model_availability(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        token: Token | None = kwargs.get("token", None)
        if not token:
            log.error("Token not found")
            return JsonResponse({"error": "Token not found"}, status=404)
        body: dict | None = kwargs.get("pydantic_model", None)
        if not body:
            return await view_func(request, *args, **kwargs)
        else:
            model: str | None = body.get("model", None)
            if not model:
                return await view_func(request, *args, **kwargs)
            else:
                if await sync_to_async(token.model_excluded)(model):
                    log.error(f"Model not found - {model}")
                    return JsonResponse({"error": "Model not found!"}, status=404)
                return await view_func(request, *args, **kwargs)

    return wrapper


def check_mcp_server_availability(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        token: Token | None = kwargs.get("token", None)
        if not token:
            log.error("Token not found")
            return JsonResponse({"error": "Token not found"}, status=404)
        server_name: str | None = kwargs.get("name", None)
        if not server_name:
            return await view_func(request, *args, **kwargs)
        else:
            if await sync_to_async(token.mcp_server_excluded)(server_name):
                log.error(f"MCP server not found - {server_name}")
                return JsonResponse({"error": "MCP server not found!"}, status=404)
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
            file_bytes = base64.b64decode(file_b64)
            return file_bytes
        except Exception as e:
            raise ValueError(f"Failed to decode base64 file data: {e}")

    elif file_id:
        # file is given as an id of a file object
        try:
            file_obj = await FileObject.objects.select_related("token__user").aget(id=file_id)
            if token and token.user != file_obj.token.user:
                raise FileObject.DoesNotExist

            file_bytes = await sync_to_async(file_obj.read)()
            return file_bytes
        except FileObject.DoesNotExist as e:
            raise e
        except Exception as e:
            raise ValueError(f"Failed to read file with id {file_id}: {e}")
    else:
        raise RuntimeError("Neither 'file_data' nor 'file_id' are given.")


def process_file_content(view_func):
    """Decorator to process file content in chat completions using Tika."""

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        token: Token | None = kwargs.get("token", None)
        pydantic_model: dict | None = kwargs.get("pydantic_model", None)
        if not pydantic_model:
            log.error("Invalid request - missing request body")
            return JsonResponse({"error": "Invalid request: missing request body"}, status=400)

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
                    file = FileFile(**content_item.get("file", {}))
                    try:
                        file_bytes = await file_to_bytes(token, file)
                    except FileObject.DoesNotExist:
                        log.error("File not found")
                        return JsonResponse({"error": "File not found"}, status=404)
                    except Exception as e:
                        # return json response here if there was an error
                        log.error(f"Error processing file - {str(e)}")
                        return JsonResponse(
                            {"error": f"Error processing file: {str(e)}"}, status=400
                        )

                    if len(file_bytes) > max_file_bytes:
                        log.error(
                            f"File processing error - File too large (individual file must be "
                            f"<= {max_file_mb}MB)"
                        )
                        return JsonResponse(
                            {
                                "error": f"Error processing file content: File too large. "
                                f"Individual file must be <= {max_file_mb}MB."
                            },
                            status=400,
                        )
                    total_file_size_bytes += len(file_bytes)
                    if total_file_size_bytes > max_total_size_bytes:
                        log.error(
                            f"File processing error - Files too large in total "
                            f"(all files must be <= {max_total_size_mb}MB)"
                        )
                        return JsonResponse(
                            {
                                "error": f"Error processing file content: Files too large in total. "
                                f"All files must be <= {max_total_size_mb}MB."
                            },
                            status=400,
                        )

                    # Extract text using Tika
                    try:
                        extracted_text = await extract_text_with_tika(file_bytes)
                    except httpx.HTTPStatusError as e:
                        # return json response here if there was a tika request error
                        log.error(f"Tika error extracting text from file - {str(e)}")
                        return JsonResponse(
                            {"error": f"Tika error extracting text from file: {str(e)}"}, status=400
                        )

                    extracted_text = (
                        f"Content of user-uploaded file '{file.get('filename', 'unknown filename')}':"
                        f"\n---\n{extracted_text}\n---"
                    )

                    # Replace file content with extracted text
                    content_item["type"] = "text"
                    content_item["text"] = extracted_text
                    del content_item["file"]

        return await view_func(request, *args, **kwargs)

    return wrapper


def catch_router_exceptions(view_func):
    def _r(e: Exception) -> str:
        s = str(e)
        s = re.sub(r"Lite-?[lL][lL][mM]", "Aqueduct", s)  # uppercase
        return re.sub(r"lite-?[lL][lL][mM]", "aqueduct", s)  # lowercase

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        # https://docs.litellm.ai/docs/exception_mapping#litellm-exceptions
        # also except equivalent openai exceptions
        try:
            return await view_func(request, *args, **kwargs)
        except (litellm.BadRequestError, openai.BadRequestError) as e:
            log.error(f"Bad request - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=400)
        except (litellm.AuthenticationError, openai.AuthenticationError) as e:
            log.error(f"Authentication error - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=401)
        except (litellm.exceptions.PermissionDeniedError, openai.PermissionDeniedError) as e:
            log.error(f"Permission denied - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=403)
        except (litellm.NotFoundError, openai.NotFoundError) as e:
            log.error(f"Not found - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=404)
        except (litellm.UnprocessableEntityError, openai.UnprocessableEntityError) as e:
            log.error(f"Unprocessable entity - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=422)
        except (litellm.RateLimitError, openai.RateLimitError) as e:
            log.error(f"Rate limit exceeded - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=429)
        except (litellm.Timeout, openai.APITimeoutError) as e:
            log.error(f"Timeout - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=504)
        except (
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            openai.APIConnectionError,
        ) as e:
            log.error(f"Service unavailable - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=503)
        except (litellm.InternalServerError, openai.InternalServerError) as e:
            log.error(f"Internal server error - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=500)
        except (litellm.APIError, openai.APIError) as e:
            # APIError is raised e.g. when user sends extra kwargs in the request body,
            # so we return a 400 Bad request.
            log.error(f"API error - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=400)
        except Exception as e:
            log.error(f"Unexpected error - {_r(e)}")
            return JsonResponse({"error": _r(e)}, status=500)

    return wrapper


def tos_accepted(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        if settings.TOS_ENABLED and settings.TOS_GATEWAY_VALIDATION:
            token: Token = kwargs.get("token")
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
                    return JsonResponse(
                        {
                            "error": "In order to use the API you have to agree to the terms of service!"
                        },
                        status=403,
                    )

        return await view_func(request, *args, **kwargs)

    return wrapper


def mcp_transport_security(view_func):
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
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        from django.conf import settings

        # Skip validation if DNS rebinding protection is disabled
        if not getattr(settings, "MCP_ENABLE_DNS_REBINDING_PROTECTION", True):
            return await view_func(request, *args, **kwargs)

        log.debug(f"MCP request headers: {dict(request.headers)}")

        # Validate Content-Type for POST requests
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            log.debug(f"POST request Content-Type: '{content_type}'")
            if not content_type.lower().startswith("application/json"):
                log.error(f"Invalid Content-Type header: {content_type}")
                return JsonResponse({"error": "Invalid Content-Type header"}, status=400)

        # Validate Host header against allowed values
        allowed_hosts = getattr(settings, "MCP_ALLOWED_HOSTS", [])
        host = request.headers.get("host")

        if not host:
            log.error("Missing Host header in request")
            return JsonResponse({"error": "Invalid Host header"}, status=421)

        host_valid = in_wildcard(host, allowed_hosts)
        if not host_valid:
            log.error(f"Invalid Host header: {host}")
            return JsonResponse({"error": "Invalid Host header"}, status=421)

        # Validate Origin header against allowed values
        # Origin can be absent for same-origin requests, so it's only validated if present
        allowed_origins = getattr(settings, "MCP_ALLOWED_ORIGINS", [])
        origin = request.headers.get("origin")
        if origin:
            origin_valid = in_wildcard(origin, allowed_origins)
            if not origin_valid:
                log.error(f"Invalid Origin header: {origin}")
                return JsonResponse({"error": "Invalid Origin header"}, status=403)

        return await view_func(request, *args, **kwargs)

    return wrapper


def parse_jsonrpc_message(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        session_id = request.headers.get("Mcp-Session-Id")
        kwargs["session_id"] = session_id

        if request.method != "POST":
            if not session_id:
                log.error(f"Session ID required for MCP server '{kwargs.get('name')}'")
                return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

            return await view_func(request, request_log=None, *args, **kwargs)

        data = kwargs["pydantic_model"]
        # For mcp requests, timeout should not be passed to the JSON RPC Message
        data.pop("timeout", None)
        json_rpc_message = JSONRPCMessage.model_validate(data)
        is_initialize = (
            hasattr(json_rpc_message.root, "method")
            and json_rpc_message.root.method == "initialize"
        )

        if not is_initialize and not session_id:
            log.error(f"Session ID required for MCP server '{kwargs.get('name')}'")
            return JsonResponse({"error": "Mcp-Session-Id header required"}, status=400)

        kwargs["json_rpc_message"] = json_rpc_message
        kwargs["is_initialize"] = is_initialize

        return await view_func(request, *args, **kwargs)

    return wrapper
