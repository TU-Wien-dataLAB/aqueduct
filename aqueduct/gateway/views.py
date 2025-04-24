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
import abc
import re  # Add re import
from typing import Optional, Callable

from django.utils import timezone
from management.models import Usage, Model, Request, Token, Endpoint, EndpointBackend

logger = logging.getLogger(__name__)


# --- Backend Abstraction ---

class AIGatewayBackend(abc.ABC):
    """
    Abstract base class for backend-specific logic within the AI Gateway.
    Subclasses handle tasks like finding the correct model and extracting usage
    based on the specifics of the target API (e.g., OpenAI, Anthropic).
    """

    @abc.abstractmethod
    def get_model(self, request: HttpRequest, endpoint: Endpoint) -> Optional[Model]:
        """
        Retrieves the Model database object based on the request data and the target endpoint.
        Returns None if the model cannot be determined or found based on the request.

        Args:
            request (HttpRequest): The incoming request.
            endpoint (Endpoint): The target Endpoint instance.

        Returns:
            Optional[Model]: The corresponding Model instance, or None.

        Raises:
            Http404: For unexpected errors during processing.
        """
        pass

    @abc.abstractmethod
    def extract_usage(self, response_body: bytes) -> Usage:
        """
        Extracts usage information (e.g., token counts) from a relayed API response body.

        Args:
            response_body (bytes): The body of the response from the relayed request.

        Returns:
            Usage: Usage extracted from the response.
        """
        pass

    @abc.abstractmethod
    def post_processing_endpoints(self) -> dict[str, list[Callable[[HttpResponse], HttpResponse]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, no regex)
        to a *list* of callable functions that perform the required post-processing transformations
        on the HttpResponse.
        """
        pass

    def requires_post_processing(self, request: HttpRequest) -> bool:
        """
        Checks if the given request's remaining path matches any path defined
        as a key in the dictionary returned by post_processing_endpoints.

        Args:
            request (HttpRequest): The incoming request.

        Returns:
            bool: True if the path matches a pattern requiring post-processing, False otherwise.
        """
        remaining_path = request.resolver_match.kwargs.get('remaining_path', '')
        # Check if the cleaned path exists as a key in the post-processing dict
        return remaining_path.lstrip('/') in self.post_processing_endpoints()


class OpenAIBackend(AIGatewayBackend):
    """
    Backend implementation for OpenAI-compatible API endpoints.
    """

    def get_model(self, request: HttpRequest, endpoint: Endpoint) -> Optional[Model]:
        """
        Extracts the model name from the request body (JSON 'model' key)
        and retrieves the Model object belonging to the specified endpoint.
        Returns None if 'model' key is missing, JSON is invalid, or model not found.
        """
        model_name = None  # Initialize model_name
        try:
            if not request.body:
                logger.warning(
                    f"OpenAIBackend.get_model called with empty request body for endpoint '{endpoint.slug}'.")
                return None  # Return None if body is empty

            data = json.loads(request.body)
            model_name = data.get('model')

            if not model_name:
                logger.warning(f"Request body missing 'model' key for endpoint '{endpoint.slug}'.")
                return None  # Return None if 'model' key is missing

            # Ensure the model belongs to the correct endpoint
            model = Model.objects.get(name=model_name, endpoint=endpoint)
            logger.debug(f"Found model '{model_name}' for endpoint '{endpoint.slug}'.")
            return model

        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON from request body for endpoint '{endpoint.slug}'.", exc_info=True)
            return None  # Return None if JSON is invalid
        except Model.DoesNotExist:
            # Use model_name captured earlier in the log message
            log_model_name = model_name if model_name else "<not provided>"
            logger.warning(f"Model with name '{log_model_name}' not found for endpoint '{endpoint.slug}'.")
            return None  # Return None if model not found
        except Exception as e:  # Catch only truly unexpected errors
            logger.error(f"Unexpected error getting model for endpoint '{endpoint.slug}': {e}", exc_info=True)
            raise Http404("Error processing request.")  # Re-raise for unexpected issues

    def extract_usage(self, response_body: bytes) -> Usage:
        """
        Extracts token usage from an OpenAI API JSON response body.
        Looks for {"usage": {"prompt_tokens": X, "completion_tokens": Y}}
        """
        try:
            data = json.loads(response_body)
            usage_dict = data.get('usage')

            if isinstance(usage_dict, dict):
                input_tokens = usage_dict.get('prompt_tokens', 0)
                output_tokens = usage_dict.get('completion_tokens', 0)
                logger.debug(
                    f"OpenAIBackend: Successfully extracted usage: Input={input_tokens}, Output={output_tokens}")
                return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
            else:
                # Check for streaming chunks which might contain usage
                # This is a simplified check; a more robust solution might parse line-by-line
                if b'"usage":' in response_body:
                    # Try to find the last usage object in potentially streamed data
                    try:
                        # Find the last occurrence of a potential usage JSON object
                        last_usage_idx = response_body.rfind(b'{"usage":')
                        if last_usage_idx != -1:
                            # Attempt to parse from that point
                            # Find the closing brace `}` for the usage object
                            end_brace_idx = response_body.find(b'}}', last_usage_idx)
                            if end_brace_idx != -1:
                                usage_json_str = response_body[last_usage_idx:end_brace_idx + 2].decode('utf-8',
                                                                                                        errors='ignore')
                                usage_data = json.loads(usage_json_str)
                                usage_dict = usage_data.get('usage')
                                if isinstance(usage_dict, dict):
                                    input_tokens = usage_dict.get('prompt_tokens', 0)
                                    output_tokens = usage_dict.get('completion_tokens', 0)
                                    logger.debug(
                                        f"OpenAIBackend: Extracted usage from streamed data: Input={input_tokens}, Output={output_tokens}")
                                    return Usage(input_tokens=input_tokens, output_tokens=output_tokens)
                    except Exception as stream_parse_e:
                        logger.warning(
                            f"Could not parse potential usage from streamed OpenAI response: {stream_parse_e}")

                logger.warning("No usable 'usage' dictionary found in OpenAI response body.")
                return Usage(input_tokens=0, output_tokens=0)

        except json.JSONDecodeError:
            logger.warning("Failed to decode JSON from OpenAI response body when extracting usage.")
            return Usage(input_tokens=0, output_tokens=0)
        except Exception as e:
            logger.error(f"Unexpected error extracting usage from OpenAI response: {e}", exc_info=True)
            return Usage(input_tokens=0, output_tokens=0)

    def post_processing_endpoints(self) -> dict[str, list[Callable[[HttpResponse], HttpResponse]]]:
        """
        Returns a dictionary of path patterns to post-processing callables for OpenAI.
        Currently, none are defined, so it returns an empty dict.
        """
        # Example: return {
        #    "chat/completions": [lambda r: step1(r), lambda r: step2(r)],
        #    "v1/other/endpoint": [lambda r: validation_step(r)]
        # }
        return {}


# --- Backend Dispatcher ---

# Map backend enum values (from models.EndpointBackend) to backend classes
BACKEND_MAP = {
    EndpointBackend.OPENAI: OpenAIBackend,
    # Add other backends here, e.g.:
    # EndpointBackend.ANTHROPIC: AnthropicBackend,
}


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

        # --- Initialize variables --- 
        request_log = None
        start_time = None
        backend_instance: Optional[AIGatewayBackend] = None

        try:
            # 2. Get Endpoint, Backend, Model, and Token
            try:
                endpoint = self.get_endpoint(request, **kwargs)
                backend_class = BACKEND_MAP.get(endpoint.backend)

                if not backend_class:
                    logger.error(
                        f"No backend implementation found for endpoint '{endpoint.slug}' with backend type '{endpoint.backend}'")
                    return JsonResponse({'error': 'Gateway configuration error: Unsupported backend'}, status=501)

                backend_instance = backend_class()  # Instantiate the backend
                model = backend_instance.get_model(request, endpoint)
                # model can be None here if not found/specified in request

            except Http404 as e:  # Catches endpoint not found or unexpected errors in get_model
                logger.warning(f"Failed to get endpoint or model for request: {e}")
                return JsonResponse({'error': str(e)}, status=404)
            except Exception as e:
                logger.error(f"Unexpected error getting endpoint/model/backend: {e}", exc_info=True)
                return JsonResponse({'error': "Internal gateway error"}, status=500)

            try:
                token = self.get_token(request)
            except Http404 as e:
                logger.error(f"Failed to get token for authenticated user {request.user.email}: {e}")
                return JsonResponse({'error': str(e)}, status=404)  # Consider 401/403 based on policy

            # 3. Create Initial Request Log Entry
            request_log = Request(
                token=token,
                model=model,  # Assign the model instance (can be None)
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
            target_api_base = endpoint.url.rstrip('/')  # Use the determined endpoint
            remaining_path_cleaned = remaining_path.lstrip('/')
            target_url = f"{target_api_base}/{remaining_path_cleaned}"

            # Set the path on the request log
            request_log.path = f"/{remaining_path_cleaned}"  # Store with leading slash for consistency

            method = request.method
            headers = {
                "Content-Type": "application/json"  # Best practice to include content type
            }
            # TODO: update model name in body to model name in endpoint
            body = request.body

            # Add/replace auth header for the target endpoint
            self.add_auth_header(headers, endpoint)  # Pass endpoint

            logger.info(
                f"Relaying {method} request for {request.user.email} (Token: {token.name}) "
                f"to {target_url} via Endpoint '{endpoint.name}'"
            )

            start_time = time.monotonic()

            # Prepare request arguments, conditionally adding 'data'
            request_args = {
                'method': method,
                'url': target_url,
                'headers': headers,
                'stream': True,
                'timeout': 60  # Consider making this configurable
            }
            if body:
                request_args['data'] = body

            relayed_response = requests.request(**request_args)
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)

            # --- Process Relayed Response --- 
            request_log.status_code = relayed_response.status_code
            request_log.response_time_ms = response_time_ms

            # Read content for usage extraction *before* creating Django response
            response_content = b"".join(relayed_response.iter_content(chunk_size=8192))

            # 5. Extract Usage (using subclass implementation)
            if model and backend_instance:  # Only extract usage if model was identified and backend exists
                try:
                    usage = backend_instance.extract_usage(response_content)
                    request_log.token_usage = usage  # Uses setter: input_tokens=..., output_tokens=...
                    logger.debug(
                        f"Request Log {request_log.id if request_log.pk else '(unsaved)'} - Extracted usage for model '{model.name}': Input={usage.input_tokens}, Output={usage.output_tokens}")
                except Exception as e:
                    logger.error(f"Error extracting usage from response for model '{model.name}': {e}", exc_info=True)
                    # Log and continue even if usage extraction fails
            else:
                # If model is None, usage is considered zero and not extracted
                request_log.input_tokens = 0
                request_log.output_tokens = 0
                logger.debug(
                    f"Request Log {request_log.id if request_log.pk else '(unsaved)'} - Model not specified or found in request, skipping usage extraction.")

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

            # Perform post-processing if required by the backend for this path
            if backend_instance and backend_instance.requires_post_processing(request):
                relative_path = remaining_path.lstrip('/')
                # Only run pipeline if the initial response was successful
                if response.status_code < 400:
                    processing_pipeline = backend_instance.post_processing_endpoints().get(relative_path)
                    if processing_pipeline:
                        logger.info(f"Running post-processing pipeline ({len(processing_pipeline)} steps) for path '{relative_path}' using {type(backend_instance).__name__}")
                        # original_response_status = response.status_code # Keep original status for logging if needed
                        for i, step_func in enumerate(processing_pipeline):
                            try:
                                response = step_func(response)
                                logger.debug(f"Post-processing step {i+1} completed. Current status: {response.status_code}")
                                # Check for error status code (4xx or 5xx) introduced by the step
                                if response.status_code >= 400:
                                    logger.warning(f"Post-processing pipeline stopped early at step {i+1} due to status code {response.status_code}.")
                                    # Error occurred, return the error response immediately
                                    return response
                            except Exception as pp_err:
                                logger.error(f"Error during post-processing step {i+1} for path '{relative_path}': {pp_err}", exc_info=True)
                                # Return a generic 500 error if a step fails unexpectedly
                                return JsonResponse({"error": f"Gateway error during response post-processing step {i+1}"}, status=500)

                        logger.debug(f"Post-processing pipeline completed successfully for path '{relative_path}'. Final status: {response.status_code}")
                else:
                    logger.debug(f"Skipping post-processing for path '{relative_path}' due to initial response status code {response.status_code}.")

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
        except Exception as e:
            logger.error(f"Unexpected error getting endpoint '{endpoint_slug}': {e}", exc_info=True)
            raise Http404("Error processing request.")

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

    def add_auth_header(self, headers: dict, endpoint: Endpoint):
        """
        Adds the Authorization header for the target endpoint to the headers dict.

        Retrieves the access token from the endpoint using get_access_token()
        and adds it as 'Authorization: Bearer <token>' if found.

        Args:
            headers (dict): The dictionary of headers to modify.
            endpoint (Endpoint): The target Endpoint object.
        """
        if not endpoint:
            logger.warning("add_auth_header called without a valid endpoint.")
            return

        access_token = endpoint.get_access_token()

        if access_token:
            # Set the Authorization header for the relayed request
            # This will overwrite any existing Authorization header from the original request
            headers['Authorization'] = f"Bearer {access_token}"
            logger.debug(f"Added Authorization header for endpoint {endpoint.name}.")
        else:
            # Log if no token was found, which might be expected or an error
            logger.warning(
                f"No access token configured or found for endpoint {endpoint.name}. Authorization header not added.")
