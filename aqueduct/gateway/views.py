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

        # 5. Extract Usage (using backend instance method - signature unchanged)
        # Check for model existence on the backend instance
        # Only attempt extraction if the relay was somewhat successful (not 502/504/500 initially)
        # and we actually got some content back.
        # Status codes like 4xx from the target API might still have usage info.
        if backend.model:
            if response.content and response.status_code not in [500, 502, 504]:
                try:
                    usage = backend.extract_usage(response.content)
                    backend.request_log.token_usage = usage  # Uses setter: input_tokens=..., output_tokens=...
                    logger.debug(
                        f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Extracted usage for model '{backend.model.display_name}': Input={usage.input_tokens}, Output={usage.output_tokens}")  # Log display_name
                except Exception as e:  # Keep this specific try-except for usage extraction
                    logger.error(f"Error extracting usage from response for model '{backend.model.display_name}': {e}",
                                 exc_info=True)  # Log display_name
                    # Log and continue even if usage extraction fails
            else:
                # If relay failed badly or no content, usage is zero
                logger.debug(
                    f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Relay status {response.status_code} or empty content, skipping usage extraction.")
                # Ensure zero usage is set if extraction skipped or failed (done in request_sync finally block now)
        else:
            logger.debug(
                f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Model not specified or found in request, skipping usage extraction.")

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
