import base64
import io
import json
import logging
import mimetypes
import os
import time
from datetime import timedelta
from functools import wraps
import re

import httpx
import litellm
import openai
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Sum
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.utils import timezone
from pydantic import ValidationError, TypeAdapter
from openai.types.chat import ChatCompletionStreamOptionsParam
from openai.types.chat.chat_completion_content_part_param import FileFile

from gateway.authentication import token_from_request
from management.models import Request, Token, Usage, FileObject

logger = logging.getLogger(__name__)


def token_authenticated(token_auth_only: bool):
    def decorator(view_func):
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args, **kwargs):
            unauthorized_response = JsonResponse({'error': 'Authentication Required'}, status=401)
            # Authentication Check
            if not (await request.auser()).is_authenticated:
                user = await auth.aauthenticate(request=request)
                if user is not None:
                    request.user = user  # Manually assign user
            else:
                request.user = await request.auser()

            if not getattr(request, "user", None) or not request.user.is_authenticated:
                logger.warning("Authentication check failed in ai_gateway_view: request.user is not authenticated.")
                return unauthorized_response
            logger.debug(f"User {request.user.email} authenticated.")

            token_key = token_from_request(request)
            if token_auth_only and not token_key:
                return unauthorized_response

            if token_key:
                token = await sync_to_async(Token.find_by_key)(token_key)
            else:
                # user is authenticated but not via token -> take first token via Token.objects to use async ORM
                token = await Token.objects.filter(user=request.user).afirst()

            if not token:
                return unauthorized_response
            kwargs['token'] = token
            return await view_func(request, *args, **kwargs)

        return wrapper

    return decorator


def parse_body(model: TypeAdapter):
    def decorator(view_func):
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args, **kwargs):
            try:
                content_type = request.headers.get("content-type", "")

                if content_type.startswith("application/json"):
                    body = request.body.decode("utf-8")
                    data = json.loads(body)
                    model.validate_python(data)

                elif content_type.startswith("multipart/form-data"):
                    data = {}
                    for key, value in request.POST.items():
                        try:
                            data[key] = json.loads(value)
                        except (TypeError, json.JSONDecodeError):
                            data[key] = value

                    max_file_bytes = settings.AQUEDUCT_FILES_API_MAX_FILE_SIZE_MB * 1024 * 1024
                    total_file_size_bytes = 0
                    for key, file in request.FILES.items():
                        data[key] = file.read()
                        if len(data[key]) > max_file_bytes:
                            return JsonResponse({'error': 'File too large'}, status=413)
                        total_file_size_bytes += len(data[key])
                        if total_file_size_bytes > 32 * 1024 * 1024:
                            return JsonResponse({'error': 'Files too large'}, status=413)

                    model.validate_python(data)

                    # Update bytes to BytesIO because pydantic TypeAdapter has problems with BytesIO.
                    # OpenAI usually expects a name for the file object (not just bytes).
                    # This only works if the field is also typed as bytes.
                    for key, file in request.FILES.items():
                        buffer: io.BytesIO = io.BytesIO(data[key])
                        buffer.name = file.name
                        data[key] = buffer
                else:
                    return JsonResponse(
                        {"error": f"Unsupported Content-Type: {content_type}"},
                        status=415
                    )

                kwargs['pydantic_model'] = data
                kwargs['pydantic_model']['timeout'] = settings.RELAY_REQUEST_TIMEOUT
                return await view_func(request, *args, **kwargs)
            except ValidationError as e:
                return JsonResponse({'error': str(e)}, status=400)

        return wrapper

    return decorator


def ensure_usage(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        model: dict | None = kwargs.get('pydantic_model', None)
        if not model:
            return await view_func(request, *args, **kwargs)
        else:
            stream = model.get('stream', False)
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
        token: Token | None = kwargs.get('token', None)
        if not token:
            return JsonResponse({'error': 'Token not found'}, status=404)

        try:
            # Get limits asynchronously
            limits = await sync_to_async(token.get_limit)()
            logger.debug(f"Rate limits for Token '{token.name}' (ID: {token.id}): {limits}")

            if limits.requests_per_minute is not None or limits.input_tokens_per_minute is not None or limits.output_tokens_per_minute is not None:
                # Define the time window for usage check (last 60 seconds)
                time_window_start = timezone.now() - timedelta(seconds=60)

                # Query recent usage asynchronously using Django's async ORM
                recent_requests_agg = await Request.objects.filter(
                    token=token,
                    timestamp__gte=time_window_start
                ).aaggregate(
                    request_count=Count('id'),
                    total_input_tokens=Sum('input_tokens'),
                    total_output_tokens=Sum('output_tokens')
                )

                request_count = recent_requests_agg.get('request_count', 0) or 0
                total_input = recent_requests_agg.get('total_input_tokens', 0) or 0
                total_output = recent_requests_agg.get('total_output_tokens', 0) or 0

                logger.debug(f"Recent usage (last 60s) for Token '{token.name}': "
                             f"Requests={request_count}, Input={total_input}, Output={total_output}")

                # --- Check Limits ---
                exceeded = []

                if limits.requests_per_minute is not None and request_count >= limits.requests_per_minute:
                    exceeded.append(f"Request limit ({limits.requests_per_minute}/min)")

                if limits.input_tokens_per_minute is not None and total_input >= limits.input_tokens_per_minute:
                    exceeded.append(f"Input token limit ({limits.input_tokens_per_minute}/min)")

                if limits.output_tokens_per_minute is not None and total_output >= limits.output_tokens_per_minute:
                    exceeded.append(f"Output token limit ({limits.output_tokens_per_minute}/min)")

                if len(exceeded) > 0:
                    error_message = "Rate limit exceeded. " + ", ".join(exceeded) + "."
                    logger.warning(
                        f"Rate limit exceeded for Token '{token.name}' (ID: {token.id}). Details: {error_message}")
                    # Return 429 Too Many Requests
                    return JsonResponse({"error": error_message}, status=429)

        except Exception as e:
            logger.error(f"Error checking rate limits for Token '{token.name}': {e}", exc_info=True)
            return JsonResponse({"error": "Internal gateway error checking rate limits"}, status=500)

        return await view_func(request, *args, **kwargs)

    return wrapper


def log_request(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        pydantic_model: dict | None = kwargs.get('pydantic_model', None)
        token = kwargs.get('token', None)
        request_log = Request(
            token=token,
            model=None if not pydantic_model else pydantic_model.get('model', None),
            # Use the resolved model from self
            timestamp=timezone.now(),
            method=request.method,
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.META.get('REMOTE_ADDR')
            # path, Status, time, usage set later in the view or processing steps
        )
        # Calculate and set path (ensure leading slash)
        request_log.path = f"/{request.path.lstrip('/')}"
        kwargs['request_log'] = request_log
        logger.debug("Initial request log object created.")
        # Note: The log is NOT saved here; it's saved later in the view after relaying.

        start_time = time.monotonic()
        result: HttpResponse | StreamingHttpResponse = await view_func(request, *args, **kwargs)
        end_time = time.monotonic()

        request_log.response_time_ms = int((end_time - start_time) * 1000)
        request_log.status_code = result.status_code

        await request_log.asave()
        return result

    return wrapper


def check_model_availability(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        token: Token | None = kwargs.get('token', None)
        if not token:
            return JsonResponse({'error': 'Token not found'}, status=404)
        body: dict | None = kwargs.get('pydantic_model', None)
        if not body:
            return await view_func(request, *args, **kwargs)
        else:
            model: str | None = body.get('model', None)
            if not model:
                return await view_func(request, *args, **kwargs)
            else:
                if await sync_to_async(token.model_excluded)(model):
                    return JsonResponse({'error': 'Model not found!'}, status=404)
                return await view_func(request, *args, **kwargs)

    return wrapper


async def extract_text_with_tika(file_bytes: bytes) -> str:
    """Extract text from file bytes using Tika API."""
    tika_url = f"{getattr(settings, 'TIKA_SERVER_URL', 'http://localhost:9998')}/tika"

    async with httpx.AsyncClient() as client:
        response = await client.put(
            tika_url,
            content=file_bytes,
            headers={},
            timeout=30.0
        )
        response.raise_for_status()
        return response.text


async def file_to_bytes(token: Token | None, file: FileFile) -> bytes:
    """Convert file description to bytes and content type."""
    file_id = file.get('file_id', None)
    file_data = file.get('file_data', None)
    filename = file.get('filename', 'unknown')

    if file_data:
        # file data contains b64 encoded files -> decode as bytes
        try:
            header, file_b64 = file_data.split(",", maxsplit=1)  # removes data uri (data:application/pdf;base64,<b64>)
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
        token: Token | None = kwargs.get('token', None)
        pydantic_model: dict | None = kwargs.get('pydantic_model', None)
        if not pydantic_model:
            return JsonResponse({'error': 'Invalid request: missing request body'}, status=400)

        messages = pydantic_model.get('messages', [])
        if not messages:
            return await view_func(request, *args, **kwargs)

        # Process messages to extract text from file content
        total_file_size_bytes = 0
        for message in messages:
            content = message.get('content', [])
            if not isinstance(content, list):
                continue

            for content_item in content:
                if isinstance(content_item, dict) and content_item.get('type') == 'file':
                    try:
                        file = FileFile(**content_item.get('file', {}))
                        file_bytes = await file_to_bytes(token, file)
                        if len(file_bytes) > 10 * 1024 * 1024:
                            return JsonResponse({
                                                    'error': 'Error processing file content: File too large. Individual file must be <= 10MB.'},
                                                status=400)

                        total_file_size_bytes += len(file_bytes)
                        if total_file_size_bytes > 32 * 1024 * 1024:
                            return JsonResponse({
                                                    'error': 'Error processing file content: Files too large in total. All files must be <= 32MB.'},
                                                status=400)

                        # Extract text using Tika
                        extracted_text = await extract_text_with_tika(file_bytes)
                        extracted_text = f"Content of user-uploaded file '{file.get('filename', 'unknown filename')}':\n---\n{extracted_text}\n---"

                        # Replace file content with extracted text
                        content_item['type'] = 'text'
                        content_item['text'] = extracted_text
                        del content_item['file']

                    except FileObject.DoesNotExist:
                        return JsonResponse({'error': "File not found"}, status=404)
                    except Exception as e:
                        # return json response here if there was an error
                        logger.info(f"Error processing file content: {e}")
                        return JsonResponse({'error': f'Error processing file: {str(e)}'}, status=400)

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
            return JsonResponse({"error": _r(e)}, status=400)
        except (litellm.AuthenticationError, openai.AuthenticationError) as e:
            return JsonResponse({"error": _r(e)}, status=401)
        except (litellm.exceptions.PermissionDeniedError, openai.PermissionDeniedError) as e:
            return JsonResponse({"error": _r(e)}, status=403)
        except (litellm.NotFoundError, openai.NotFoundError) as e:
            return JsonResponse({"error": _r(e)}, status=404)
        except (litellm.UnprocessableEntityError, openai.UnprocessableEntityError) as e:
            return JsonResponse({"error": _r(e)}, status=422)
        except (litellm.RateLimitError, openai.RateLimitError) as e:
            return JsonResponse({"error": _r(e)}, status=429)
        except (litellm.Timeout, openai.APITimeoutError) as e:
            return JsonResponse({"error": _r(e)}, status=504)
        except litellm.ServiceUnavailableError as e:
            return JsonResponse({"error": _r(e)}, status=503)
        except (litellm.InternalServerError, openai.InternalServerError) as e:
            return JsonResponse({"error": _r(e)}, status=500)
        except (litellm.APIConnectionError, litellm.APIError, openai.APIConnectionError, openai.APIError) as e:
            return JsonResponse({"error": _r(e)}, status=500)
        except Exception as e:
            return JsonResponse({"error": _r(e)}, status=500)

    return wrapper
