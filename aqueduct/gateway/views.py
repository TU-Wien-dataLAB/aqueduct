# gateway/views.py
from django.contrib import auth
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse, HttpRequest, Http404
from django.views import View
from django.views.decorators.csrf import csrf_exempt
import logging
import requests  # Added for relaying requests
from requests import Request as RequestsRequest # Alias to avoid confusion with models.Request
from requests import Session as RequestsSession # Alias for Session
import time
from typing import Optional
import re # Added import

from django.utils import timezone

from gateway.backends.base import AIGatewayBackend
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
                endpoint=endpoint,
                timestamp=timezone.now(),
                method=request.method,
                user_agent=request.headers.get('User-Agent', ''),
                ip_address=request.META.get('REMOTE_ADDR')
                # path, Status, time, usage set later
            )

            # --- Prepare arguments for requests.Request --- 
            remaining_path = kwargs.get('remaining_path', '')
            # Ensure leading/trailing slashes are handled correctly for joining
            target_api_base = endpoint.url.rstrip('/')
            remaining_path_cleaned = remaining_path.lstrip('/')
            target_url = f"{target_api_base}/{remaining_path_cleaned}"
            request_log.path = f"/{remaining_path_cleaned}" # Store path with leading slash

            # Prepare initial headers and body for the outbound request
            outbound_headers = {
                "Content-Type": request.content_type or "application/json", # Use original content-type or default
                # Copy other relevant headers? Be careful not to leak internal info.
                # Example: "Accept": request.headers.get("Accept", "*/*"),
            }
            # Add/replace auth header for the target endpoint
            self.add_auth_header(outbound_headers, endpoint)

            # Get body from original request
            outbound_body = request.body

            # Create the requests.Request object
            request_object = RequestsRequest(
                method=request.method,
                url=target_url,
                headers=outbound_headers,
                data=outbound_body
            )

            # --- Pre-processing Pipeline (operates on request_object) --- 
            if backend_instance and backend_instance.requires_pre_processing(request):
                relative_path = remaining_path.lstrip('/')
                pre_processing_pipeline = []
                matched_patterns = []
                for pattern, pipeline in backend_instance.pre_processing_endpoints().items():
                    if re.fullmatch(pattern, relative_path):
                        pre_processing_pipeline.extend(pipeline)
                        matched_patterns.append(pattern)

                if pre_processing_pipeline:
                    logger.info(
                        f"Running combined pre-processing pipeline ({len(pre_processing_pipeline)} steps) for path '{relative_path}' (matched patterns: {matched_patterns}) using {type(backend_instance).__name__}")
                    for i, step_func in enumerate(pre_processing_pipeline):
                        try:
                            # Pass the current request_object, expect Request or HttpResponse back
                            result = step_func(backend_instance, request, request_log, request_object, None)

                            if isinstance(result, HttpResponse):
                                logger.warning(
                                    f"Pre-processing pipeline returned an HttpResponse at step {i + 1} (Status: {result.status_code}). Short-circuiting relay.")
                                request_log.status_code = result.status_code
                                try:
                                    request_log.save()
                                    logger.debug(f"Saved Request log entry ID (pre-processing short-circuit): {request_log.id}")
                                except Exception as save_err:
                                    logger.error(f"Failed to save Request log during pre-processing short-circuit: {save_err}", exc_info=True)
                                return result # Return the short-circuit response

                            elif isinstance(result, RequestsRequest):
                                # Success, update request_object for the next step
                                request_object = result
                                logger.debug(f"Pre-processing step {i + 1} completed successfully, request object updated.")
                            else:
                                # Should not happen based on type hints, but handle defensively
                                logger.error(
                                    f"Pre-processing step {i + 1} for path '{relative_path}' returned an unexpected type: {type(result)}. Aborting.")
                                request_log.status_code = 500
                                try: request_log.save() # Attempt to save log
                                except Exception: pass
                                return JsonResponse({"error": f"Internal gateway error during request pre-processing step {i + 1}"}, status=500)

                        except Exception as pre_err:
                            logger.error(f"Error during pre-processing step {i + 1} for path '{relative_path}': {pre_err}", exc_info=True)
                            request_log.status_code = 500
                            try: request_log.save()
                            except Exception as save_err:
                                logger.error(f"Failed to save Request log during pre-processing error handling: {save_err}", exc_info=True)
                            return JsonResponse({"error": f"Internal gateway error during request pre-processing step {i + 1}"}, status=500)

                    logger.debug(f"Pre-processing pipeline completed successfully for path '{relative_path}'.")

            # 4. Prepare and Relay Request using requests.Session
            logger.info(
                f"Relaying {request_object.method} request for {request.user.email} (Token: {token.name}) "
                f"to {request_object.url} via Endpoint '{endpoint.name}'"
            )

            start_time = time.monotonic()

            # Use a session to prepare and send the request
            with RequestsSession() as session:
                # Prepare the request (handles headers, body encoding etc.)
                prepared_request = session.prepare_request(request_object)

                # Log final headers being sent (optional, be careful with sensitive info)
                # logger.debug(f"Prepared request headers: {prepared_request.headers}")

                relayed_response = session.send(
                    prepared_request,
                    stream=True,
                    timeout=60 # Consider making this configurable
                )

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
            logger.debug(f"Saved Request log entry ID: {request_log.id} for {request_object.method} {request_object.url}")

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
                    processing_pipeline = [] # Initialize as empty list
                    matched_patterns = [] # Keep track of patterns that matched
                    # Find all matching pipelines and extend
                    for pattern, pipeline in backend_instance.post_processing_endpoints().items():
                        if re.fullmatch(pattern, relative_path):
                            processing_pipeline.extend(pipeline)
                            matched_patterns.append(pattern)
                            # Removed break; continue checking other patterns

                    if processing_pipeline:
                        logger.info(
                            f"Running combined post-processing pipeline ({len(processing_pipeline)} steps) for path '{relative_path}' (matched patterns: {matched_patterns}) using {type(backend_instance).__name__}")
                        # original_response_status = response.status_code # Keep original status for logging if needed
                        for i, step_func in enumerate(processing_pipeline):
                            try:
                                response = step_func(backend_instance, request, request_log, response)
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
                        f"Skipping post-processing for path '{relative_path}' (status code {response.status_code}), no matching pattern found or required.")

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
