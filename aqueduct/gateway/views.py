# gateway/views.py
import json
import time
from datetime import timedelta
from functools import wraps
from typing import AsyncGenerator, Any

import litellm
import openai
from asgiref.sync import async_to_sync, sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Sum, Count
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import logging

from django.utils import timezone
from django.views.decorators.http import require_POST, require_GET
from litellm import Router, TextCompletionStreamWrapper
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse, TextCompletionResponse, EmbeddingResponse
from openai.types.chat import ChatCompletionStreamOptionsParam
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import Type, TypedDict

from gateway.authentication import token_from_request
from management.models import Request, Token, Usage
from gateway.router import get_router, get_router_config

logger = logging.getLogger(__name__)

COMPLETION_TYPE = openai.types.Completion
CHAT_COMPLETION_TYPE = openai.types.chat.ChatCompletion
STREAMING_CHAT_COMPLETION_TYPE = openai.types.chat.ChatCompletionChunk


# TODO: check auth only on token backend (import token backend and call authenticate)
def token_authenticated(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        # Authentication Check
        if not (await request.auser()).is_authenticated:
            user = await auth.aauthenticate(request=request)
            if user is not None:
                request.user = user  # Manually assign user
        else:
            request.user = await request.auser()

        if not getattr(request, "user", None) or not request.user.is_authenticated:
            logger.warning("Authentication check failed in ai_gateway_view: request.user is not authenticated.")
            return JsonResponse(
                {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided and valid.'},
                status=401
            )
        logger.debug(f"User {request.user.email} authenticated.")

        token_key = token_from_request(request)
        token = await sync_to_async(Token.find_by_key)(token_key)
        kwargs['token'] = token
        return await view_func(request, *args, **kwargs)

    return wrapper


def parse_body(model: TypeAdapter):
    def decorator(view_func):
        @wraps(view_func)
        async def wrapper(request: ASGIRequest, *args, **kwargs):
            try:
                body = request.body.decode('utf-8')
                model.validate_json(body)
                kwargs['pydantic_model'] = json.loads(body)
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
        token_key = token_from_request(request)
        token = await sync_to_async(Token.find_by_key)(token_key)
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


def _usage_from_bytes(content: bytes) -> Usage:
    try:
        usage_dict = json.loads(content).get('usage', None)
        if isinstance(usage_dict, dict):
            # CompletionUsage(completion_tokens=70, prompt_tokens=35, total_tokens=105, completion_tokens_details=None, prompt_tokens_details=None)
            return Usage(
                input_tokens=usage_dict.get('prompt_tokens', 0),
                output_tokens=usage_dict.get('completion_tokens', 0)
            )
        else:
            return Usage(input_tokens=0, output_tokens=0)
    except json.JSONDecodeError:
        return Usage(input_tokens=0, output_tokens=0)


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


def handle_timeout(view_func):
    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        try:
            return await view_func(request, *args, **kwargs)
        except litellm.Timeout as e:
            return JsonResponse({"error": str(e)}, status=504)

    return wrapper


def _openai_stream(completion: CustomStreamWrapper | TextCompletionStreamWrapper, request_log: Request) -> \
        AsyncGenerator[str, Any]:
    start_time = time.monotonic()

    async def stream():
        token_usage = Usage(0, 0)
        async for chunk in completion:
            chunk_str = chunk.model_dump_json(exclude_none=True, exclude_unset=True)
            token_usage += _usage_from_bytes(chunk_str.encode('utf-8'))
            try:
                yield f"data: {chunk_str}\n\n"
            except Exception as e:
                yield f"data: {str(e)}\n\n"

        request_log.token_usage = token_usage
        request_log.response_time_ms = int((time.monotonic() - start_time) * 1000)
        await request_log.asave()
        # Streaming is done, yield the [DONE] chunk
        yield f"data: [DONE]\n\n"

    return stream()


@csrf_exempt
@require_POST
@token_authenticated
@check_limits
@parse_body(model=TypeAdapter(openai.types.CompletionCreateParams))
@ensure_usage
@log_request
@handle_timeout
async def completions(request: ASGIRequest, pydantic_model: openai.types.CompletionCreateParams, request_log: Request,
                      *args, **kwargs):
    router = get_router()
    completion: TextCompletionResponse | TextCompletionStreamWrapper = await router.atext_completion(**pydantic_model)
    if isinstance(completion, TextCompletionStreamWrapper):
        return StreamingHttpResponse(streaming_content=_openai_stream(completion=completion, request_log=request_log),
                                     headers={'Content-Type': 'text/event-stream'})
    elif isinstance(completion, TextCompletionResponse):
        data = completion.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _usage_from_bytes(json.dumps(data).encode("utf-8"))
        return JsonResponse(data=completion.model_dump(), status=200)
    else:
        raise NotImplementedError(f"Completion for response type {type(completion)} is not implemented.")


@csrf_exempt
@require_POST
@token_authenticated
@check_limits
@parse_body(model=TypeAdapter(openai.types.chat.CompletionCreateParams))
@ensure_usage
@log_request
@handle_timeout
async def chat_completions(request: ASGIRequest, pydantic_model: openai.types.chat.CompletionCreateParams,
                           request_log: Request, *args, **kwargs):
    router = get_router()
    chat_completion: CustomStreamWrapper | ModelResponse = await router.acompletion(**pydantic_model)
    if isinstance(chat_completion, CustomStreamWrapper):
        return StreamingHttpResponse(
            streaming_content=_openai_stream(completion=chat_completion, request_log=request_log),
            content_type='text/event-stream')
    elif isinstance(chat_completion, ModelResponse):
        data = chat_completion.model_dump(exclude_none=True, exclude_unset=True)
        request_log.token_usage = _usage_from_bytes(json.dumps(data).encode("utf-8"))
        return JsonResponse(data=data, status=200)
    else:
        raise NotImplementedError(f"Completion for response type {type(chat_completion)} is not implemented.")


@csrf_exempt
@require_POST
@token_authenticated
@check_limits
@parse_body(model=TypeAdapter(openai.types.EmbeddingCreateParams))
@ensure_usage
@log_request
@handle_timeout
async def embeddings(request: ASGIRequest, pydantic_model: openai.types.EmbeddingCreateParams, request_log: Request,
                     *args, **kwargs):
    router = get_router()
    embedding: EmbeddingResponse = await router.aembedding(**pydantic_model)
    data = embedding.model_dump(exclude_none=True, exclude_unset=True)
    request_log.token_usage = _usage_from_bytes(json.dumps(data).encode("utf-8"))
    return JsonResponse(data=data, status=200)


MODEL_CREATION_TIMESTAMP = int(timezone.now().timestamp())


@csrf_exempt
@require_GET
@token_authenticated
@log_request
async def models(request: ASGIRequest, *args, **kwargs):
    router_config = get_router_config()
    model_list: list[dict] = router_config["model_list"]

    return JsonResponse(data=dict(
        data=[
            {
                "id": model["model_name"],
                "object": "model",
                "created": MODEL_CREATION_TIMESTAMP,
                "owned_by": "aqueduct",
            }
            for model in model_list
        ],
        object="list",
    ))
