import json
import logging
import time
from datetime import timedelta
from functools import wraps
import re

import litellm
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib import auth
from django.core.handlers.asgi import ASGIRequest
from django.db.models import Count, Sum
from django.http import JsonResponse, HttpResponse, StreamingHttpResponse
from django.utils import timezone
from pydantic import ValidationError, TypeAdapter
from openai.types.chat import ChatCompletionStreamOptionsParam

from gateway.authentication import token_from_request
from management.models import Request, Token, Usage

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


def catch_router_exceptions(view_func):
    def _r(e: Exception) -> str:
        s = str(e)
        s = re.sub(r"Lite-?[lL][lL][mM]", "Aqueduct", s) # uppercase
        return re.sub(r"lite-?[lL][lL][mM]", "aqueduct", s) # lowercase

    @wraps(view_func)
    async def wrapper(request: ASGIRequest, *args, **kwargs):
        # https://docs.litellm.ai/docs/exception_mapping#litellm-exceptions
        try:
            return await view_func(request, *args, **kwargs)
        except litellm.BadRequestError as e:
            return JsonResponse({"error": _r(e)}, status=400)
        except litellm.AuthenticationError as e:
            return JsonResponse({"error": _r(e)}, status=401)
        except litellm.exceptions.PermissionDeniedError as e:
            return JsonResponse({"error": _r(e)}, status=403)
        except litellm.NotFoundError as e:
            return JsonResponse({"error": _r(e)}, status=404)
        except litellm.UnprocessableEntityError as e:
            return JsonResponse({"error": _r(e)}, status=422)
        except litellm.RateLimitError as e:
            return JsonResponse({"error": _r(e)}, status=429)
        except (litellm.APIConnectionError, litellm.APIError) as e:
            return JsonResponse({"error": _r(e)}, status=500)
        except litellm.Timeout as e:
            return JsonResponse({"error": _r(e)}, status=504)
        except litellm.ServiceUnavailableError as e:
            return JsonResponse({"error": _r(e)}, status=503)
        except litellm.InternalServerError as e:
            return JsonResponse({"error": _r(e)}, status=500)
        except Exception as e:
            return JsonResponse({"error": _r(e)}, status=500)

    return wrapper
