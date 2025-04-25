# gateway/views.py
from django.contrib import auth
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse, HttpRequest, Http404
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import logging
from typing import Optional
import re  # Added import

from django.utils import timezone

from gateway.backends.base import AIGatewayBackend, PreProcessingPipelineError
from gateway.backends.openai import OpenAIBackend
from management.models import Request, Token, Endpoint, EndpointBackend

logger = logging.getLogger(__name__)

# --- Backend Dispatcher ---

# Map backend enum values (from models.EndpointBackend) to backend classes
BACKEND_MAP = {
    EndpointBackend.OPENAI: OpenAIBackend,
    # Add other backends here, e.g.:
    # EndpointBackend.ANTHROPIC: AnthropicBackend,
}


# --- Base Gateway View with Authentication ---

class AIGatewayView(View):
    """
    Base class for AI Gateway views providing authentication.
    Relies on Django's authentication middleware (configured with a custom backend)
    to populate request.user based on the Authorization header.
    If authentication fails, dispatch returns a 401 error response.
    """
    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    @csrf_exempt
    def dispatch(self, request, *args, **kwargs):
        """
        Handles incoming requests:
        1. Checks authentication.
        2. Retrieves model and token using subclass methods.
        3. Creates a request log entry.
        4. Relays the request to the target endpoint.
        5. Updates the request log with response info (status, time, usage).
        6. Returns the response from the target or an error response.
        """
        # 1. Authentication Check
        if not request.user.is_authenticated:
            user = auth.authenticate(request=request)
            if user is not None:
                request.user = user  # Manually assign user

        if not request.user.is_authenticated:
            logger.warning("Authentication check failed in dispatch: request.user is not authenticated.")
            return JsonResponse(
                {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided and valid.'},
                status=401
            )
        logger.debug(f"User {request.user.email} authenticated.")

        try:
            # 2. Get Endpoint, Backend, Model, and Token (and run pre-processing)
            endpoint = self.get_endpoint(request, **kwargs)
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
            backend: AIGatewayBackend = backend_class(request, endpoint)

        except Http404 as e:
            logger.warning(f"Failed to initialize backend or resolve model/token: {e}")
            return JsonResponse({'error': str(e)}, status=404)
        except PreProcessingPipelineError as e:
            logger.info(
                f"Pre-processing pipeline short-circuited at step {e.step_index} with status {e.response.status_code}")
            return e.response

        # 4. Relay Request using Backend Method
        response = backend.request_sync()

        return response

    def get_endpoint(self, request: HttpRequest, **kwargs) -> Endpoint:
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
            endpoint = Endpoint.objects.get(slug=endpoint_slug)
            logger.debug(f"Found endpoint '{endpoint_slug}' for request.")
            return endpoint
        except Endpoint.DoesNotExist:
            logger.warning(f"Endpoint with slug '{endpoint_slug}' not found.")
            raise Http404(f"Endpoint '{endpoint_slug}' not found.")
