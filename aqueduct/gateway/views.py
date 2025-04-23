# gateway/views.py
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
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

class AIGatewayView(LoginRequiredMixin, View):
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

        authenticated = False
        if request.user.is_authenticated:
            authenticated = True
        else:
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

        # --- Initialize variables --- 
        request_log = None
        start_time = None

        try:
            # 2. Get Model and Token (using subclass implementations)
            try:
                model = self.get_model(request)
            except Http404 as e:
                logger.warning(f"Failed to get model for request: {e}")
                return JsonResponse({'error': str(e)}, status=404)

            try:
                token = self.get_token(request)
            except Http404 as e:
                logger.error(f"Failed to get token for authenticated user {request.user.email}: {e}")
                return JsonResponse({'error': str(e)}, status=404)  # Consider 401/403 based on policy

            # 3. Create Initial Request Log Entry
            request_log = Request(
                token=token,
                model=model,
                timestamp=timezone.now(),
                method=request.method,
                user_agent=request.headers.get('User-Agent', ''),
                ip_address=request.META.get('REMOTE_ADDR')
                # Status, time, usage set later
            )
            # request_log.save() # Optionally save early, but saving after response is better

            # 4. Prepare and Relay Request
            remaining_path = kwargs.get('remaining_path', '')
            # Ensure leading/trailing slashes are handled correctly for joining
            target_api_base = model.endpoint.url.rstrip('/')
            remaining_path_cleaned = remaining_path.lstrip('/')
            target_url = f"{target_api_base}/{remaining_path_cleaned}"

            # Set the path on the request log
            request_log.path = f"/{remaining_path_cleaned}" # Store with leading slash for consistency

            method = request.method
            headers = dict(request.headers)
            body = request.body

            # Clean up headers for relaying
            headers.pop('Host', None)
            headers.pop('Cookie', None)  # Avoid leaking session info
            # Add/replace auth header for the target endpoint
            self.add_auth_header(headers, model)

            logger.info(
                f"Relaying {method} request for {request.user.email} (Token: {token.name}) "
                f"to {target_url} via Endpoint '{model.endpoint.name}'"
            )

            start_time = time.monotonic()
            relayed_response = requests.request(
                method=method,
                url=target_url,
                headers=headers,
                data=body,
                stream=True,  # Important for handling large responses / streaming
                timeout=60  # Consider making this configurable
            )
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)

            # --- Process Relayed Response --- 
            request_log.status_code = relayed_response.status_code
            request_log.response_time_ms = response_time_ms

            # Read content for usage extraction *before* creating Django response
            response_content = b"".join(relayed_response.iter_content(chunk_size=8192))

            # 5. Extract Usage (using subclass implementation)
            try:
                usage = self.extract_usage(response_content)
                request_log.token_usage = usage  # Uses setter: input_tokens=..., output_tokens=...
                logger.debug(
                    f"Request Log {request_log.id if request_log.pk else '(unsaved)'} - Extracted usage: Input={usage.input_tokens}, Output={usage.output_tokens}")
            except Exception as e:
                logger.error(f"Error extracting usage from response: {e}", exc_info=True)
                # Decide if failure to extract usage is critical. Here, we log and continue.

            request_log.save()  # Save successful request log details
            logger.debug(f"Saved Request log entry ID: {request_log.id} for {method} {target_url}")

            # 6. Construct and Return Django Response
            response = HttpResponse(
                content=response_content,
                status=relayed_response.status_code,
                content_type=relayed_response.headers.get('Content-Type')
            )

            # Copy relevant headers from relayed response to Django response
            hop_by_hop_headers = [
                'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
                'te', 'trailers', 'transfer-encoding', 'upgrade'
            ]
            for key, value in relayed_response.headers.items():
                if key.lower() not in hop_by_hop_headers:
                    response[key] = value

            logger.debug(f"Relay successful: Status {relayed_response.status_code}")
            return response

        # --- Error Handling for Relaying --- 
        except requests.exceptions.Timeout:
            end_time = time.monotonic()
            logger.warning(f"Gateway timeout relaying request to {target_url}")
            if request_log:  # Log if request_log object exists
                request_log.status_code = 504  # Gateway Timeout
                if start_time:  # Check if timer started
                    request_log.response_time_ms = int((end_time - start_time) * 1000)
                request_log.save()
                logger.debug(f"Saved Request log entry ID (timeout): {request_log.id}")
            return JsonResponse({"error": "Gateway timeout"}, status=504)

        except requests.exceptions.RequestException as e:
            end_time = time.monotonic()
            logger.error(f"Error relaying request to {target_url}: {e}", exc_info=True)
            if request_log:  # Log if request_log object exists
                # Try to get status from target response if available
                status_code = 502  # Bad Gateway default
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                request_log.status_code = status_code
                if start_time:
                    request_log.response_time_ms = int((end_time - start_time) * 1000)
                request_log.save()
                logger.debug(f"Saved Request log entry ID (relay error): {request_log.id}")
            return JsonResponse({"error": "Gateway error during relay"}, status=502)

        except Exception as e:
            # Catch any other unexpected errors during the process
            end_time = time.monotonic()
            logger.error(f"Unexpected error during gateway dispatch: {e}", exc_info=True)
            if request_log:  # Log if request_log object exists
                request_log.status_code = 500  # Internal Server Error
                if start_time:
                    request_log.response_time_ms = int((end_time - start_time) * 1000)
                # Ensure usage isn't accidentally non-zero if error occurred before extraction
                if request_log.input_tokens is None: request_log.input_tokens = 0
                if request_log.output_tokens is None: request_log.output_tokens = 0
                request_log.save()
                logger.debug(f"Saved Request log entry ID (unexpected error): {request_log.id}")
            # Return generic error to client
            return JsonResponse({"error": "Internal gateway error"}, status=500)
        finally:
            # Final check to ensure logging if an error happened *after* request_log creation
            # but *before* any explicit save or *between* save and return.
            if request_log and not request_log.pk:
                try:
                    if request_log.status_code is None: request_log.status_code = 500
                    if request_log.input_tokens is None: request_log.input_tokens = 0
                    if request_log.output_tokens is None: request_log.output_tokens = 0
                    # Add response time if possible, otherwise leave as null/default
                    if start_time and 'end_time' in locals():  # Check if end_time was set
                        request_log.response_time_ms = int((end_time - start_time) * 1000)
                    request_log.save()
                    logger.warning(f"Saved Request log entry ID in finally block: {request_log.id}")
                except Exception as final_save_e:
                    logger.error(f"Failed to save Request log in finally block: {final_save_e}", exc_info=True)

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
    Implements OpenAI-specific logic for model lookup and usage extraction.
    The actual request relaying and logging is handled by the base AIGatewayView.dispatch.
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
