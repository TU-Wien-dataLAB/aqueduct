# gateway/views.py
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed, HttpRequest, Http404
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import requests  # Added for relaying requests
import json  # Added for parsing request body
import time

from django.utils import timezone
from management.models import Usage, Model, Request, Token

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

    def get_token(self, request: HttpRequest) -> Token:
        """
        Retrieves the Token object associated with the current request.

        Priority:
        1. Looks for 'Authorization: Bearer <token>' header.
        2. If header not found or invalid, falls back to the first token
           associated with the authenticated request.user.

        Args:
            request (HttpRequest): The incoming request.

        Returns:
            Token: The corresponding Token instance.

        Raises:
            Http404: If no token can be found via header or associated user.
        """
        token = None
        auth_header = request.headers.get('Authorization')

        if auth_header and auth_header.startswith('Bearer '):
            try:
                token_key = auth_header.split(' ')[1]
                if token_key:
                    token = Token.find_by_key(token_key)
            except IndexError:
                logger.warning("Could not parse Bearer token from Authorization header.")
            except Exception as e:
                logger.error(f"Error looking up token from header: {e}", exc_info=True)

        # Fallback: If no token from header and user is authenticated
        if token is None and request.user and request.user.is_authenticated:
            logger.debug(f"No valid Bearer token in header, attempting to find token for user {request.user.email}")
            # Fetch the first available token for this user.
            # Use select_related if you often need the user/SA details immediately.
            # Order by pk or created_at to get a consistent 'first' token.
            token = request.user.custom_auth_tokens.order_by('pk').first()

        if token is None:
            logger.error("Could not associate request with a token (checked header and user tokens).")
            raise Http404("Unable to determine authentication token for request.")

        logger.debug(f"Associated request with Token ID {token.id} (Name: {token.name})")
        return token

    def add_auth_header(self, headers: dict, model: Model):
        """
        Adds the Authorization header for the target endpoint to the headers dict.

        Retrieves the access token from the model's endpoint using get_access_token()
        and adds it as 'Authorization: Bearer <token>' if found.

        Args:
            headers (dict): The dictionary of headers to modify.
            model (Model): The Model object, used to find the target Endpoint.
        """
        if not model or not model.endpoint:
            logger.warning("add_auth_header called without a valid model or endpoint.")
            return

        access_token = model.endpoint.get_access_token()

        if access_token:
            # Set the Authorization header for the relayed request
            # This will overwrite any existing Authorization header from the original request
            headers['Authorization'] = f"Bearer {access_token}"
            logger.debug(f"Added Authorization header for endpoint {model.endpoint.name}.")
        else:
            # Log if no token was found, which might be expected or an error
            logger.warning(
                f"No access token configured or found for endpoint {model.endpoint.name}. Authorization header not added.")

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

    def extract_usage(self, response_body: bytes) -> Usage:
        """
        Extracts token usage from an OpenAI API JSON response body.

        Looks for {"usage": {"prompt_tokens": X, "completion_tokens": Y}}

        Args:
            response_body (bytes): The raw response body from the OpenAI API.

        Returns:
            Usage: A dataclass containing the extracted input and output tokens.
                   Returns Usage(0, 0) if parsing fails or usage info is missing.
        """
        try:
            data = json.loads(response_body)
            usage_dict = data.get('usage')

            if isinstance(usage_dict, dict):
                input_tokens = usage_dict.get('prompt_tokens', 0)
                output_tokens = usage_dict.get('completion_tokens', 0)
                logger.debug(f"Successfully extracted usage: Input={input_tokens}, Output={output_tokens}")
                return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            else:
                logger.warning("No 'usage' dictionary found in OpenAI response body or it's not a dict.")
                return Usage(input_tokens=0, output_tokens=0)

        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from OpenAI response body when extracting usage.")
            return Usage(input_tokens=0, output_tokens=0)
        except Exception as e:
            logger.error(f"Unexpected error extracting usage from OpenAI response: {e}", exc_info=True)
            return Usage(input_tokens=0, output_tokens=0)

    def dispatch(self, request, *args, **kwargs):
        """
        Handles all HTTP methods for paths starting with '/v1/'.
        1. Calls the base class dispatch to handle authentication.
        2. If authenticated, relays the request to the target backend service.
        3. Returns the response from the backend service or an error response.
        """
        # 1. Let the base class handle authentication.
        auth_response = super().dispatch(request, *args, **kwargs)
        if isinstance(auth_response, HttpResponse):
            return auth_response  # Authentication failed or handled by base class

        # Get the associated Model
        try:
            model = self.get_model(request)
        except Http404 as e:
            # Log the error and return appropriate response
            # No Request object created yet as model lookup failed
            return JsonResponse({'error': str(e)}, status=404)

        # --- Get Associated Token ---
        try:
            token = self.get_token(request)
        except Http404 as e:
            # Log the error and return appropriate response
            logger.error(f"Failed to get token for authenticated user {request.user.email}")
            return JsonResponse({'error': str(e)}, status=404)  # Or 401/403 depending on policy
        # --- End Token Retrieval ---

        # --- Create Request Log Entry ---
        request_log = Request(
            token=token,  # Use the token found by get_token
            model=model,
            timestamp=timezone.now(),
            method=request.method,
            user_agent=request.headers.get('User-Agent', ''),
            ip_address=request.META.get('REMOTE_ADDR')
            # status_code, response_time_ms, input/output tokens will be set later
        )
        # --- End Request Log Entry Creation ---

        # 2. Authentication successful. Proceed with relaying.
        remaining_path = kwargs.get('remaining_path', '')
        target_api_base = model.endpoint.url
        target_url = f"{target_api_base}/v1/{remaining_path}"

        method = request.method
        headers = dict(request.headers)
        body = request.body

        # --- TODO: Refine headers for relaying ---
        headers.pop('Host', None)
        headers.pop('Cookie', None)
        # Add authentication header for the target endpoint
        self.add_auth_header(headers, model)
        # --- End TODO ---

        logger.info(
            f"Relaying {method} request for {request.user.email} "
            f"to {target_url} (Path: v1/{remaining_path})"
        )

        start_time = time.monotonic()
        try:
            # 3. Make the relayed request
            relayed_response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                stream=True,
                timeout=60
            )
            end_time = time.monotonic()

            # --- Update Request Log with Response Info ---
            request_log.status_code = relayed_response.status_code
            request_log.response_time_ms = int((end_time - start_time) * 1000)
            # --- End Update ---

            # Read content for usage extraction *before* creating HttpResponse
            # Using iter_content ensures we handle streamed responses correctly
            # We need the full content to attempt parsing usage later
            response_content = b"".join(relayed_response.iter_content(chunk_size=8192))

            # --- Extract Usage ---
            try:
                usage = self.extract_usage(response_content)
                request_log.token_usage = usage  # Use the setter on the Request model
                logger.debug(
                    f"Updated request log with usage: Input={usage.input_tokens}, Output={usage.output_tokens}")
            except Exception as e:
                # Log error during usage extraction, but don't fail the request
                logger.error(f"Error extracting usage from response or updating log: {e}", exc_info=True)
            # --- End Usage Extraction ---

            request_log.save()  # Save successful request log
            logger.debug(f"Saved Request log entry ID: {request_log.id}")

            # 4. Construct the Django response from the relayed response
            response = HttpResponse(
                content=response_content,  # Use the content we already read
                status=relayed_response.status_code,
                content_type=relayed_response.headers.get('Content-Type')
            )

            # --- Copy relevant headers from relayed response ---
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
            end_time = time.monotonic()
            logger.warning(f"Gateway timeout relaying request to {target_url}")
            # --- Update Request Log on Timeout ---
            request_log.status_code = 504  # Gateway Timeout
            request_log.response_time_ms = int((end_time - start_time) * 1000)
            request_log.save()
            logger.debug(f"Saved Request log entry ID (timeout): {request_log.id}")
            # --- End Update ---
            return JsonResponse({"error": "Gateway timeout"}, status=504)
        except requests.exceptions.RequestException as e:
            end_time = time.monotonic()
            logger.error(f"Error relaying request to {target_url}: {e}", exc_info=True)
            # --- Update Request Log on Relay Error ---
            request_log.status_code = 502  # Bad Gateway (standard for relay errors)
            if hasattr(e, 'response') and e.response is not None:
                request_log.status_code = e.response.status_code  # Or use target's status if available
            request_log.response_time_ms = int((end_time - start_time) * 1000)
            request_log.save()
            logger.debug(f"Saved Request log entry ID (relay error): {request_log.id}")
            # --- End Update ---
            return JsonResponse({"error": "Gateway error during relay"}, status=502)
        except Exception as e:
            end_time = time.monotonic()
            # Catch any other unexpected errors during relay
            logger.error(f"Unexpected error during relay to {target_url}: {e}", exc_info=True)
            # --- Update Request Log on Unexpected Error ---
            request_log.status_code = 500  # Internal Server Error
            # Response time might be inaccurate if error occurred early
            if start_time:  # Check if start_time was set
                request_log.response_time_ms = int((end_time - start_time) * 1000)
            request_log.save()
            logger.debug(f"Saved Request log entry ID (unexpected error): {request_log.id}")
            # --- End Update ---
            return JsonResponse({"error": "Internal gateway error"}, status=500)
        finally:
            # Ensure the request log is saved even if an unhandled exception occurs after its creation
            # but before the explicit save calls in try/except blocks.
            # This might result in duplicate saves if an error occurs *after* a save
            # but before the function returns, but ensures logging in most cases.
            # A more robust solution might involve transaction management.
            if request_log and not request_log.pk:  # Only save if not already saved
                try:
                    # Populate with generic error code if not set
                    if request_log.status_code is None:
                        request_log.status_code = 500  # Default to internal error
                    request_log.save()
                    logger.warning(f"Saved Request log entry ID in finally block: {request_log.id}")
                except Exception as final_save_e:
                    logger.error(f"Failed to save Request log in finally block: {final_save_e}", exc_info=True)
