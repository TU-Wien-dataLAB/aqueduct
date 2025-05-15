# gateway/views.py
import httpx
from asgiref.sync import async_to_sync, sync_to_async
from django.contrib import auth
from django.http import HttpResponse, JsonResponse, HttpRequest, Http404
from django.views.decorators.csrf import csrf_exempt
import logging
from typing import Optional
import re  # Added import

from django.utils import timezone

from gateway.backends.base import AIGatewayBackend, PreProcessingPipelineError
from gateway.backends.openai import OpenAIBackend
from management.models import Request, Token, Endpoint, EndpointBackend

logger = logging.getLogger(__name__)

async_client = httpx.AsyncClient(timeout=60, follow_redirects=True)

# --- Backend Dispatcher ---

# Map backend enum values (from models.EndpointBackend) to backend classes
BACKEND_MAP = {
    EndpointBackend.OPENAI: OpenAIBackend,
    # Add other backends here, e.g.:
    # EndpointBackend.ANTHROPIC: AnthropicBackend,
}


async def get_endpoint(request: HttpRequest, **kwargs) -> Endpoint:
    """
    Retrieves the Endpoint database object based on the slug from the URL.

    Args:
        request (HttpRequest): The incoming request.
        **kwargs: Keyword arguments captured from the URL pattern.

    Returns:
        Endpoint: The corresponding Endpoint instance.

    Raises:
        Http404: If the 'endpoint_slug' is missing or no matching endpoint is found.
    """
    endpoint_slug = kwargs.get('endpoint_slug')
    if not endpoint_slug:
        logger.error("Endpoint slug missing from URL kwargs.")
        raise Http404("Gateway URL configuration error.")

    try:
        endpoint = await Endpoint.objects.aget(slug=endpoint_slug)
        logger.debug(f"Found endpoint '{endpoint_slug}' for request.")
        return endpoint
    except Endpoint.DoesNotExist:
        logger.warning(f"Endpoint with slug '{endpoint_slug}' not found.")
        raise Http404(f"Endpoint '{endpoint_slug}' not found.")


@csrf_exempt
async def ai_gateway_view(request: HttpRequest, *args, **kwargs):
    """
    Functional view for AI Gateway endpoint.
    Handles:
    1. Authentication.
    2. Endpoint, backend, model, and token resolution.
    3. Request log creation.
    4. Relaying the request to the target endpoint.
    5. Updating the request log with response info.
    6. Returning the response from the target or an error response.
    """
    # 1. Authentication Check
    if not getattr(request, "user", None) or not request.user.is_authenticated:
        user = await auth.aauthenticate(request=request)
        if user is not None:
            request.user = user  # Manually assign user

    if not getattr(request, "user", None) or not request.user.is_authenticated:
        logger.warning("Authentication check failed in ai_gateway_view: request.user is not authenticated.")
        return JsonResponse(
            {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided and valid.'},
            status=401
        )
    logger.debug(f"User {request.user.email} authenticated.")

    try:
        # 2. Get Endpoint, Backend, Model, and Token (and run pre-processing)
        endpoint = await get_endpoint(request, **kwargs)
        backend_class = BACKEND_MAP.get(endpoint.backend)

        if not backend_class:
            logger.error(
                f"No backend implementation found for endpoint '{endpoint.slug}' with backend type '{endpoint.backend}'")
            return JsonResponse({'error': 'Gateway configuration error: Unsupported backend'}, status=501)

        # Instantiate the backend. This now handles:
        # - Resolving token (raises Http404)
        # - Resolving model (raises Http404)
        # - Preparing relay request
        # - Running pre-processing (raises BackendProcessingError or PreProcessingPipelineError)
        backend: AIGatewayBackend = backend_class(request, endpoint, async_client)
        await backend.initialize()

    except Http404 as e:
        logger.warning(f"Failed to initialize backend or resolve model/token: {e}")
        return JsonResponse({'error': str(e)}, status=404)
    except PreProcessingPipelineError as e:
        logger.info(
            f"Pre-processing pipeline short-circuited at step {e.step_index} with status {e.response.status_code}")
        return e.response

    # 4. Relay Request using Backend Method
    if backend.is_streaming_request():
        response = await backend.request_streaming()
    else:
        response = await backend.request_non_streaming()

    return response
