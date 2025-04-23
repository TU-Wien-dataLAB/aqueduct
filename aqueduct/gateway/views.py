# gateway/views.py
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed, HttpRequest, Http404
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import requests  # Added for relaying requests
import json  # Added for parsing request body

from management.models import Usage, Model

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

    def get_model(self, request: HttpRequest) -> Model:
        """
        Retrieves the Model database object based on the request data.
        This method must be implemented by subclasses.

        Args:
            request (HttpRequest): The incoming request.

        Returns:
            Model: The corresponding Model instance.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("get_model() must be implemented in a subclass.")

    def extract_usage(self, response_body: bytes) -> Usage:
        """
        Dummy method to extract usage from a relayed API response.

        Args:
            response_body (bytes): The body of the response from the relayed request.

        Returns:
            Usage: Usage extracted from the response.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.
        """
        raise NotImplementedError("extract_usage() must be implemented in a subclass.")


class V1OpenAIGateway(AIGatewayView):
    """
    Handles API requests prefixed with 'v1/'. Inherits authentication from AIGatewayView.
    Relays authenticated requests to a configured backend service using the dispatch method.
    """

    def get_model(self, request: HttpRequest) -> Model:
        """
        Extracts the model name from the request body and retrieves the Model object.

        Args:
            request (HttpRequest): The incoming request, expected to have a JSON body
                                 with a 'model' key.

        Returns:
            Model: The matching Model instance.

        Raises:
            Http404: If the request body is invalid JSON, missing the 'model' key,
                     or the specified model name does not exist in the database.
        """
        if not request.body:
            logger.warning("get_model called with empty request body.")
            raise Http404("Request body is empty.")

        try:
            data = json.loads(request.body)
            model_name = data.get('model')

            if not model_name:
                logger.warning("Request body missing 'model' key.")
                raise Http404("Request body must contain a 'model' key.")

            model = Model.objects.get(name=model_name)
            logger.debug(f"Found model '{model_name}' for request.")
            return model

        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from request body.", exc_info=True)
            raise Http404("Invalid JSON in request body.")
        except Model.DoesNotExist:
            logger.warning(f"Model with name '{model_name}' not found.")
            raise Http404(f"Model '{model_name}' not found.")
        except Exception as e:  # Catch unexpected errors
            logger.error(f"Unexpected error getting model: {e}", exc_info=True)
            raise Http404("Error processing request.")  # Generic error for security

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

        model = self.get_model(request)

        # 2. Authentication successful. Proceed with relaying.
        # request.user is guaranteed to be an authenticated User instance here.
        remaining_path = kwargs.get('remaining_path', '')
        target_api_base = model.endpoint.url
        target_url = f"{target_api_base}/v1/{remaining_path}"

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
                timeout=60  # Set an appropriate timeout
            )

            # 4. Construct the Django response from the relayed response
            response = HttpResponse(
                content=relayed_response.raw,  # Use .raw for streaming content
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
            return JsonResponse({"error": "Gateway error during relay"}, status=502)  # Bad Gateway
        except Exception as e:
            # Catch any other unexpected errors during relay
            logger.error(f"Unexpected error during relay to {target_url}: {e}", exc_info=True)
            return JsonResponse({"error": "Internal gateway error"}, status=500)

    # No longer need specific methods like post(), get(), etc.
    # All requests are handled by dispatch after authentication.
