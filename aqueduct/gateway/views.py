# gateway/views.py
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import requests  # Added for relaying requests

logger = logging.getLogger(__name__)


# --- Base Gateway View with Authentication ---

@method_decorator(csrf_exempt, name='dispatch')
class AIGatewayView(View):
    """
    Base class for AI Gateway views providing authentication.
    Relies on Django's authentication middleware (configured with a custom backend)
    to populate request.user based on the Authorization header.
    If authentication fails, dispatch returns a 401 error response.
    """
    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def dispatch(self, request, *args, **kwargs):
        """
        Checks if the user was successfully authenticated by the middleware.
        If not authenticated, returns a 401 Unauthorized response.
        If authenticated, calls the standard Django View dispatch mechanism,
        which will route the request to the appropriate HTTP method handler
        (e.g., get(), post()) on the subclass.
        """
        # Authentication middleware runs before dispatch and should set request.user
        if not request.user.is_authenticated:
            # User was not authenticated by any configured backend
            logger.warning("Authentication check failed in dispatch: request.user is not authenticated.")
            return JsonResponse(
                {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided and valid.'},
                status=401
            )

        # Authentication successful - proceed with standard view dispatching
        logger.debug(f"User {request.user.email} authenticated. Proceeding with dispatch.")
        return super().dispatch(request, *args, **kwargs)


# --- V1 OpenAI Specific Gateway View ---

class V1OpenAIGateway(AIGatewayView):
    """
    Handles API requests prefixed with 'v1/'. Inherits authentication from AIGatewayView.
    Relays authenticated requests to a configured backend service using the dispatch method.
    """

    def dispatch(self, request, *args, **kwargs):
        """
        Handles all HTTP methods for paths starting with '/v1/'.
        1. Calls the base class dispatch to handle authentication.
        2. If authenticated, relays the request to the target backend service.
        3. Returns the response from the backend service or an error response.
        """
        # 1. Let the base class handle authentication.
        # If it returns an HttpResponse (e.g., 401), authentication failed or
        # the base class handled the request entirely. Return that response.
        auth_response = super().dispatch(request, *args, **kwargs)
        if isinstance(auth_response, HttpResponse):
            return auth_response

        # 2. Authentication successful. Proceed with relaying.
        # request.user is guaranteed to be an authenticated User instance here.
        remaining_path = kwargs.get('remaining_path', '')
        # --- TODO: Configure your target API endpoint ---
        # This should likely come from Django settings or environment variables
        target_api_base = "https://api.openai.com" # Example: Replace with your actual target
        target_url = f"{target_api_base}/v1/{remaining_path}"
        # --- End TODO ---

        method = request.method
        headers = dict(request.headers)
        body = request.body

        # --- TODO: Refine headers for relaying ---
        # - Remove Django-specific headers (e.g., 'Host', 'Cookie')
        # - Add any necessary headers for the target service (e.g., API keys)
        # - Ensure 'Content-Length' is correct if you modify the body
        headers.pop('Host', None)
        headers.pop('Cookie', None)
        # Example: Add target API key (fetch securely, e.g., from user profile or settings)
        # headers['Authorization'] = f"Bearer {YOUR_TARGET_API_KEY}"
        # --- End TODO ---

        logger.info(
            f"Relaying {method} request for {request.user.email} "
            f"to {target_url} (Path: v1/{remaining_path})"
        )

        try:
            # 3. Make the relayed request
            # Using stream=True is generally recommended for gateways to avoid
            # loading large request/response bodies into memory entirely.
            relayed_response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                stream=True,
                timeout=60 # Set an appropriate timeout
            )

            # 4. Construct the Django response from the relayed response
            response = HttpResponse(
                content=relayed_response.raw, # Use .raw for streaming content
                status=relayed_response.status_code,
                content_type=relayed_response.headers.get('Content-Type')
            )

            # --- Copy relevant headers from relayed response ---
            # Avoid copying hop-by-hop headers like 'Connection', 'Transfer-Encoding'
            hop_by_hop_headers = ['connection', 'keep-alive', 'proxy-authenticate',
                                  'proxy-authorization', 'te', 'trailers',
                                  'transfer-encoding', 'upgrade']
            for key, value in relayed_response.headers.items():
                if key.lower() not in hop_by_hop_headers:
                    response[key] = value
            # --- End Header Copy ---

            logger.debug(f"Relay successful: Status {relayed_response.status_code}")
            return response

        except requests.exceptions.Timeout:
            logger.warning(f"Gateway timeout relaying request to {target_url}")
            return JsonResponse({"error": "Gateway timeout"}, status=504)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error relaying request to {target_url}: {e}", exc_info=True)
            return JsonResponse({"error": "Gateway error during relay"}, status=502) # Bad Gateway
        except Exception as e:
            # Catch any other unexpected errors during relay
            logger.error(f"Unexpected error during relay to {target_url}: {e}", exc_info=True)
            return JsonResponse({"error": "Internal gateway error"}, status=500)

    # No longer need specific methods like post(), get(), etc.
    # All requests are handled by dispatch after authentication.

# --- End V1 OpenAI Specific Gateway View ---
