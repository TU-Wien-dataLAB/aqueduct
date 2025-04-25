# gateway/views.py
from django.contrib import auth
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse, HttpRequest, Http404
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import logging
from typing import Optional
import re # Added import

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

        # --- Initialize variables ---
        backend: Optional[AIGatewayBackend] = None

        try:
            # 2. Get Endpoint, Backend, Model, and Token (and run pre-processing)
            try:
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
                backend = backend_class(request, endpoint)

            except Http404 as e:  # Catches endpoint not found or errors in backend _resolve_model/_resolve_token
                logger.warning(f"Failed to initialize backend or resolve model/token: {e}")
                return JsonResponse({'error': str(e)}, status=404)
            except PreProcessingPipelineError as e:
                logger.info(f"Pre-processing pipeline short-circuited at step {e.step_index} with status {e.response.status_code}")
                # The exception carries the HttpResponse to be returned directly.
                # The log should have been saved by the step itself before raising.
                return e.response
            except Exception as e:
                logger.error(f"Unexpected error during backend initialization or processing: {e}", exc_info=True)
                # Attempt to save log if backend and log exist but failed later in init
                if 'backend' in locals() and backend and backend.request_log and not backend.request_log.pk:
                    try:
                        backend.request_log.status_code = 500
                        backend.request_log.save()
                        logger.debug(f"Saved Request log ID {backend.request_log.id} after init error")
                    except Exception as save_err:
                        logger.error(f"Failed to save Request log during backend init error handling: {save_err}", exc_info=True)
                return JsonResponse({'error': "Internal gateway error during setup"}, status=500)


            # --- Pre-processing Pipeline --- REMOVED (Now done in backend.__init__)
            # The logic is moved to backend._run_pre_processing_pipeline()

            # 4. Relay Request using Backend Method
            response = backend.request_sync()

            # 5. Extract Usage (using backend instance method - signature unchanged)
            # Check for model existence on the backend instance
            # Only attempt extraction if the relay was somewhat successful (not 502/504/500 initially)
            # and we actually got some content back.
            # Status codes like 4xx from the target API might still have usage info.
            if backend.model and backend:
                if response.content and response.status_code not in [500, 502, 504]:
                    try:
                        usage = backend.extract_usage(response.content)
                        backend.request_log.token_usage = usage  # Uses setter: input_tokens=..., output_tokens=...
                        logger.debug(
                            f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Extracted usage for model '{backend.model.display_name}': Input={usage.input_tokens}, Output={usage.output_tokens}") # Log display_name
                    except Exception as e:
                        logger.error(f"Error extracting usage from response for model '{backend.model.display_name}': {e}", exc_info=True) # Log display_name
                        # Log and continue even if usage extraction fails
                else:
                    # If relay failed badly or no content, usage is zero
                    logger.debug(
                        f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Relay status {response.status_code} or empty content, skipping usage extraction.")
                    backend.request_log.input_tokens = 0
                    backend.request_log.output_tokens = 0
            else:
                # If model is None, usage is considered zero and not extracted
                backend.request_log.input_tokens = 0
                backend.request_log.output_tokens = 0
                logger.debug(
                    f"Request Log {backend.request_log.id if backend.request_log.pk else '(unsaved)'} - Model not specified or found in request, skipping usage extraction.")

            backend.request_log.save()  # Save successful request log details
            logger.debug(f"Saved Request log entry ID: {backend.request_log.id} after relay status {response.status_code}")

            # 6. Post-Process and Return Response
            # The 'response' object is now the HttpResponse/JsonResponse from request_sync

            # Perform post-processing if required by the backend for this path (no request arg needed)
            if backend and backend.requires_post_processing():
                relative_path = kwargs.get('remaining_path', '') # Keep for logging/pattern matching
                # Only run pipeline if the initial response was successful
                if response.status_code < 400:
                    processing_pipeline = [] # Initialize as empty list
                    matched_patterns = [] # Keep track of patterns that matched
                    # Find all matching pipelines and extend
                    for pattern, pipeline in backend.post_processing_endpoints().items():
                        if re.fullmatch(pattern, relative_path):
                            processing_pipeline.extend(pipeline)
                            matched_patterns.append(pattern)
                            # Removed break; continue checking other patterns

                    if processing_pipeline:
                        logger.info(
                            f"Running combined post-processing pipeline ({len(processing_pipeline)} steps) for path '{relative_path}' (matched patterns: {matched_patterns}) using {type(backend).__name__}")
                        # original_response_status = response.status_code # Keep original status for logging if needed
                        for i, step_func in enumerate(processing_pipeline):
                            try:
                                response = step_func(backend, backend.request_log, response)
                                logger.debug(
                                    f"Post-processing step {i + 1} completed. Current status: {response.status_code}")
                                # Check for error status code (4xx or 5xx) introduced by the step
                                if response.status_code >= 400:
                                    logger.warning(
                                        f"Post-processing pipeline stopped early at step {i + 1} (patterns {matched_patterns}) due to status code {response.status_code}.")
                                    # Error occurred, return the error response immediately
                                    return response
                            except Exception as pp_err:
                                logger.error(
                                    f"Error during post-processing step {i + 1} for path '{relative_path}' (patterns {matched_patterns}): {pp_err}",
                                    exc_info=True)
                                # Return a generic 500 error if a step fails unexpectedly
                                return JsonResponse(
                                    {"error": f"Gateway error during response post-processing step {i + 1}"},
                                    status=500)

                        logger.debug(
                            f"Combined post-processing pipeline completed successfully for path '{relative_path}' (patterns {matched_patterns}). Final status: {response.status_code}")
                else:
                    logger.debug(
                        f"Skipping post-processing for path '{relative_path}' (status code {response.status_code}), because initial response was not successful.")
            else:
                # Log if post-processing was not required or backend was None
                logger.debug(f"Post-processing not required for path '{kwargs.get('remaining_path', '').lstrip('/')}'")

            return response

        # --- Error Handling for Dispatch (excluding relay errors handled by request_sync) ---
        except Exception as e:
            # Catch any other unexpected errors during the dispatch process (e.g., post-processing)
            # Note: Timeout/RequestException from relaying are handled *inside* request_sync
            # and reflected in the returned status_code.
            # Errors during backend init or pre-processing are caught earlier.
            # This mainly catches errors during usage extraction, post-processing, or response creation.
            current_time = timezone.now() # Use timezone.now() if start_time might not be set
            logger.error(f"Unexpected error during gateway dispatch (after relay attempt): {e}", exc_info=True)
            if backend.request_log:  # Log if request_log object exists
                if backend.request_log.status_code is None or backend.request_log.status_code < 400: # Avoid overwriting relay error status
                    backend.request_log.status_code = 500  # Internal Server Error if not already set to an error
                # Ensure usage isn't accidentally non-zero if error occurred before extraction
                if backend.request_log.input_tokens is None: backend.request_log.input_tokens = 0
                if backend.request_log.output_tokens is None: backend.request_log.output_tokens = 0
                backend.request_log.save()
                logger.warning(f"Saved Request log entry ID in finally block: {backend.request_log.id}")
            # Return generic error to client
            return JsonResponse({"error": "Internal gateway error"}, status=500)
        finally:
            # Final check to ensure logging if an error happened *after* request_log creation
            # but *before* any explicit save or *between* save and return.
            if backend.request_log and not backend.request_log.pk:
                try:
                    if backend.request_log.status_code is None: backend.request_log.status_code = 500
                    if backend.request_log.input_tokens is None: backend.request_log.input_tokens = 0
                    if backend.request_log.output_tokens is None: backend.request_log.output_tokens = 0
                    # Add response time if possible, otherwise leave as null/default
                    backend.request_log.save()
                    logger.warning(f"Saved Request log entry ID in finally block: {backend.request_log.id}")
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
