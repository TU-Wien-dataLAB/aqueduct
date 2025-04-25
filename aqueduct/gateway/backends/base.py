import abc
import re
from typing import Optional, Callable, Union, Tuple
import logging
import time

from django.http import HttpRequest, HttpResponse, Http404, HttpResponseServerError, JsonResponse
from django.utils import timezone
import requests
from requests import Request as RequestsRequest, Session as RequestsSession

from management.models import Endpoint, Model, Usage, Request, Token

logger = logging.getLogger(__name__)


# --- Custom Exceptions ---

class PreProcessingPipelineError(Exception):
    def __init__(self, response: HttpResponse, step_index: Optional[int] = None):
        self.response = response
        self.step_index = step_index
        super().__init__(str(response.content))


class AIGatewayBackend(abc.ABC):
    """
    Abstract base class for backend-specific logic within the AI Gateway.
    Subclasses handle tasks like finding the correct model and extracting usage
    based on the specifics of the target API (e.g., OpenAI, Anthropic).

    Instances are initialized with the Django HttpRequest and the resolved Endpoint.
    The associated Model instance (or None) is resolved during initialization
    and stored in self.model.
    The associated Token and Request log are also resolved/created and stored.
    """

    def __init__(self, request: HttpRequest, endpoint: Endpoint):
        """
        Initializes the backend instance, resolves the Model and Token,
        creates the initial Request log entry, prepares the relay request,
        and runs the pre-processing pipeline.

        Args:
            request (HttpRequest): The incoming Django request.
            endpoint (Endpoint): The resolved target Endpoint instance.

        Raises:
            Http404: If the Token or requested Model cannot be resolved.
            PreprocessingError: If an error occurs during the pre-processing pipeline.
            ValueError: If request or endpoint is invalid.
        """
        if not request or not endpoint:
            # Should not happen if instantiated correctly in the view
            raise ValueError("AIGatewayBackend requires a valid HttpRequest and Endpoint for initialization.")
        self.request: HttpRequest = request
        self.endpoint: Endpoint = endpoint
        self.model: Optional[Model] = self._resolve_model()  # Resolve and store model
        self.token: Token = self._resolve_token()  # Resolve and store token (raises Http404 if not found)
        self.request_log: Request = self._create_request_log()  # Create initial log entry
        self.relay_request: RequestsRequest = self._prepare_relay_request()  # Prepare the outbound request

        # Run pre-processing immediately after preparing the request
        self._run_pre_processing_pipeline()

    @abc.abstractmethod
    def _resolve_model(self) -> Optional[Model]:
        """
        Resolves the Model database object based on the request data (self.request)
        and the target endpoint (self.endpoint).

        This method is called during initialization.

        - If the request does not specify a model or model information is not applicable
          (e.g., for a non-inference endpoint), this method should return None.
        - If the request specifies a model that is valid and configured for the endpoint,
          it should return the corresponding Model instance.
        - If the request specifies a model that is invalid, not found, or not allowed
          for the endpoint, it should raise an appropriate exception (e.g., Http404).
        - It may also raise exceptions for other unexpected errors during processing.

        Returns:
            Optional[Model]: The corresponding Model instance if found and applicable, otherwise None.

        Raises:
            Http404: If the requested model is not found or allowed, or for other request processing errors.
            Exception: For unexpected errors during processing.
        """
        pass

    def _resolve_token(self) -> Token:
        """
        Retrieves the Token object associated with the current request (self.request).

        Priority:
        1. Looks for 'Authorization: Bearer <token>' header.
        2. If header not found or invalid, falls back to the first token
           associated with the authenticated request.user.

        Returns:
            Token: The corresponding Token instance.

        Raises:
            Http404: If no token can be found via header or associated user.
        """
        token = None
        auth_header = self.request.headers.get('Authorization')

        if auth_header and auth_header.startswith('Bearer '):
            try:
                token_key = auth_header.split(' ')[1]
                if token_key:
                    # Use the class method directly for finding the token
                    token = Token.find_by_key(token_key)
            except IndexError:
                logger.warning("Could not parse Bearer token from Authorization header.")
            except Exception as e:
                logger.error(f"Error looking up token from header: {e}", exc_info=True)

        # Fallback: If no token from header and user is authenticated
        user = getattr(self.request, 'user', None)  # Safely get user
        if token is None and user and user.is_authenticated:
            logger.debug(f"No valid Bearer token in header, attempting to find token for user {user.email}")
            # Fetch the first available token for this user.
            token = user.custom_auth_tokens.order_by('pk').first()

        if token is None:
            logger.error("Could not associate request with a token (checked header and user tokens).")
            # Raising Http404 here will be caught by the view during backend instantiation
            raise Http404("Unable to determine authentication token for request.")

        logger.debug(f"Associated request with Token ID {token.id} (Name: {token.name})")
        return token

    def _create_request_log(self) -> Request:
        """
        Creates the initial Request log entry using resolved information.
        """
        request_log = Request(
            token=self.token,
            model=self.model,  # Use the resolved model from self
            endpoint=self.endpoint,
            timestamp=timezone.now(),
            method=self.request.method,
            user_agent=self.request.headers.get('User-Agent', ''),
            ip_address=self.request.META.get('REMOTE_ADDR')
            # path, Status, time, usage set later in the view or processing steps
        )
        # Calculate and set path (ensure leading slash)
        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        request_log.path = f"/{remaining_path.lstrip('/')}"
        logger.debug("Initial request log object created.")
        # Note: The log is NOT saved here; it's saved later in the view after relaying.
        return request_log

    def add_auth_header(self, headers: dict):
        """
        Adds the Authorization header for the target endpoint to the headers dict.

        Retrieves the access token from the endpoint using get_access_token()
        and adds it as 'Authorization: Bearer <token>' if found.

        Modifies the headers dict in place.

        Args:
            headers (dict): The dictionary of headers to modify.
        """
        access_token = self.endpoint.get_access_token()

        if access_token:
            # Set the Authorization header for the relayed request
            # This will overwrite any existing Authorization header from the original request
            headers['Authorization'] = f"Bearer {access_token}"
            logger.debug(f"Added Authorization header for endpoint {self.endpoint.name}.")
        else:
            # Log if no token was found, which might be expected or an error
            logger.warning(
                f"No access token configured or found for endpoint {self.endpoint.name}. Authorization header not added.")

    def _prepare_relay_request(self) -> RequestsRequest:
        """
        Prepares the initial requests.Request object for relaying.
        This includes setting the URL, method, headers (including auth), and body.
        """
        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        # Ensure leading/trailing slashes are handled correctly for joining
        target_api_base = self.endpoint.url.rstrip('/')
        remaining_path_cleaned = remaining_path.lstrip('/')
        target_url = f"{target_api_base}/{remaining_path_cleaned}"

        # Prepare initial headers and body for the outbound request
        outbound_headers = {
            "Content-Type": self.request.content_type or "application/json",  # Use original content-type or default
            # Copy other relevant headers? Be careful not to leak internal info.
            # Example: "Accept": request.headers.get("Accept", "*/*"),
        }
        # Add/replace auth header for the target endpoint
        self.add_auth_header(outbound_headers)  # Call the method now part of the backend

        # Get body from original request
        outbound_body = self.request.body

        # Create the requests.Request object
        request_object = RequestsRequest(
            method=self.request.method,
            url=target_url,
            headers=outbound_headers,
            data=outbound_body
        )
        logger.debug(f"Prepared initial relay request: {request_object.method} {request_object.url}")
        return request_object

    def request_sync(self, timeout: int = 60) -> HttpResponse:
        """
        Synchronously sends the prepared relay request (self.relay_request)
        and handles basic connection errors and timeouts.

        Args:
            timeout (int): Timeout in seconds for the request. Default 60.

        Returns:
            response (HttpResponse | JsonResponse): The Django response object (HttpResponse for success, JsonResponse for gateway errors like 502/504).
        """
        start_time = time.monotonic()
        target_url = self.relay_request.url
        logger.info(
            f"Relaying {self.relay_request.method} request from {self.request.user.email} (Token: {self.token.name}) "
            f"to {target_url} via Endpoint '{self.endpoint.name}'"
        )
        response = None
        response_time_ms = None
        try:
            with RequestsSession() as session:
                # Prepare the request using the session (handles connection pooling, etc.)
                prepared_request = session.prepare_request(self.relay_request)

                # logger.debug(f"Prepared request headers being sent: {prepared_request.headers}")

                relayed_response = session.send(
                    prepared_request,
                    stream=True,  # Read content below, but stream helps with large responses
                    timeout=timeout
                )

            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)

            # Read content fully
            response_content = b"".join(relayed_response.iter_content(chunk_size=8192))

            # Construct Django HttpResponse
            response = HttpResponse(
                content=response_content,
                status=relayed_response.status_code,
                content_type=relayed_response.headers.get('Content-Type')
            )

            # Copy relevant headers from relayed response to Django response
            hop_by_hop_headers = [
                'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'content-encoding',
                'te', 'trailers', 'transfer-encoding', 'upgrade'
            ]
            for key, value in relayed_response.headers.items():
                if key.lower() not in hop_by_hop_headers:
                    response[key] = value

            logger.debug(f"Relay to {target_url} completed: Status {response.status_code}, Time {response_time_ms}ms")
            return response

        except requests.exceptions.Timeout:
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            logger.warning(f"Gateway timeout after {response_time_ms}ms relaying request to {target_url}")
            response = JsonResponse({"error": "Gateway timeout"}, status=504)
            return response  # Gateway Timeout

        except requests.exceptions.RequestException as e:
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            status_code = 502  # Bad Gateway default
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code  # Use target's status if available
            logger.error(
                f"Error relaying request to {target_url} (Status: {status_code}, Time: {response_time_ms}ms): {e}",
                exc_info=False)  # exc_info=False to avoid overly verbose logs for common connection errors
            response = JsonResponse({"error": "Gateway error during relay"}, status=status_code)
            return response
        finally:
            if response:
                self.request_log.status_code = response.status_code
                if response_time_ms:
                    self.request_log.response_time_ms = response_time_ms
            self.request_log.save()

    @abc.abstractmethod
    def extract_usage(self, response_body: bytes) -> Usage:
        """
        Extracts usage information (e.g., token counts) from a relayed API response body.
        This method typically does not need self.request or self.endpoint, but they
        are available if needed by a subclass. Access the resolved model via `self.model`.

        Args:
            response_body (bytes): The body of the response from the relayed request.

        Returns:
            Usage: Usage extracted from the response.
        """
        pass

    @abc.abstractmethod
    def pre_processing_endpoints(self) -> dict[str, list[Callable[
        ['AIGatewayBackend', 'Request'], Union[requests.Request, HttpResponse]]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, regex)
        to a *list* of callable functions that perform pre-processing transformations
        on the `requests.Request` object before it's sent.

        Each callable in the list receives:
        - The backend instance (self)
        - The original Django HttpRequest
        - The Request model instance (log entry)
        - The `requests.Request` object prepared for the outbound call (backend.relay_request).

        Each callable must return either:
        - A `requests.Request` object (potentially modified): Indicates success, processing continues with the returned object.
        - An HttpResponse: To short-circuit the process and return this response immediately (e.g., for validation errors).

        Note: Callables should operate on the *copy* of the request passed to them
              and return the modified version, rather than modifying in place,
              unless intended behavior guarantees no side effects on shared state.
              The view logic handles updating `backend.relay_request`.
        """
        pass

    def requires_pre_processing(self) -> bool:
        """
        Checks if the current request's (self.request) remaining path matches any
        regex pattern defined as a key in the dictionary returned by pre_processing_endpoints.

        Returns:
            bool: True if the path matches a pattern requiring pre-processing, False otherwise.
        """
        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        relative_path = remaining_path.lstrip('/')
        for pattern in self.pre_processing_endpoints():
            if re.fullmatch(pattern, relative_path):
                return True
        return False

    @abc.abstractmethod
    def post_processing_endpoints(self) -> dict[
        str, list[Callable[['AIGatewayBackend', 'Request', HttpResponse], HttpResponse]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, regex)
        to a *list* of callable functions that perform the required post-processing transformations
        on the HttpResponse.

        Each callable in the list receives:
        - The backend instance (self)
        - The original HttpRequest
        - The Request model instance (log entry)
        - The current HttpResponse (output of the previous step or initial response)

        Each callable must return the (potentially modified) HttpResponse.
        """
        pass

    def requires_post_processing(self) -> bool:
        """
        Checks if the current request's (self.request) remaining path matches any
        regex pattern defined as a key in the dictionary returned by post_processing_endpoints.

        Returns:
            bool: True if the path matches a pattern requiring post-processing, False otherwise.
        """
        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        relative_path = remaining_path.lstrip('/')
        for pattern in self.post_processing_endpoints():
            if re.fullmatch(pattern, relative_path):
                return True
        return False

    def _run_pre_processing_pipeline(self):
        """
        Executes the pre-processing pipeline based on matching endpoint patterns.
        Modifies `self.relay_request` in place.

        Raises:
            PreprocessingError: If any step returns an invalid type or raises an exception.
            HttpResponse: If a step explicitly returns an HttpResponse to short-circuit.
                          (This should be caught and returned by the caller, e.g., the view)
        """
        if not self.requires_pre_processing():
            logger.debug("Pre-processing not required for this request path.")
            return

        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        relative_path = remaining_path.lstrip('/')
        pre_processing_pipeline = []
        matched_patterns = []
        for pattern, pipeline in self.pre_processing_endpoints().items():
            if re.fullmatch(pattern, relative_path):
                pre_processing_pipeline.extend(pipeline)
                matched_patterns.append(pattern)

        if not pre_processing_pipeline:
            # Should not happen if requires_pre_processing was true, but check defensively
            logger.warning(f"requires_pre_processing was true for path '{relative_path}' but no pipeline found.")
            return

        logger.info(
            f"Running combined pre-processing pipeline ({len(pre_processing_pipeline)} steps) for path '{relative_path}' "
            f"(matched patterns: {matched_patterns}) using {type(self).__name__}")

        for i, step_func in enumerate(pre_processing_pipeline):
            step_index = i + 1
            try:
                # Pass the backend instance and request log
                result = step_func(self, self.request_log)

                if isinstance(result, HttpResponse):
                    # A step wants to short-circuit with a specific response.
                    logger.warning(
                        f"Pre-processing pipeline returned an HttpResponse at step {step_index} "
                        f"(Status: {result.status_code}). Short-circuiting relay.")
                    # Update log status before raising
                    self.request_log.status_code = result.status_code
                    try:
                        self.request_log.save()
                        logger.debug(f"Saved Request log entry ID {self.request_log.id} (pre-processing short-circuit)")
                    except Exception as save_err:
                        logger.error(f"Failed to save Request log during pre-processing short-circuit: {save_err}",
                                     exc_info=True)
                    # Raise the specific exception wrapping the response
                    raise PreProcessingPipelineError(response=result, step_index=step_index)

                elif isinstance(result, RequestsRequest):
                    # Success, update the backend's request instance for the next step or final relay
                    self.relay_request = result
                    logger.debug(f"Pre-processing step {step_index} completed successfully, relay_request updated.")
                else:
                    # Should not happen based on type hints, but handle defensively
                    logger.error(
                        f"Pre-processing step {step_index} for path '{relative_path}' returned an unexpected type: {type(result)}. Aborting.")
                    self.request_log.status_code = 500
                    try:
                        self.request_log.save()
                    except Exception:
                        pass
                    # Raise custom exception
                    error_response = JsonResponse({'error': 'Internal gateway error during request pre-processing'},
                                                  status=500)
                    raise PreProcessingPipelineError(response=error_response, step_index=step_index)

            except Exception as pre_err:
                # Catch exceptions raised *by* the step function, or the PreProcessingPipelineError raised above
                if isinstance(pre_err, PreProcessingPipelineError):
                    # Re-raise the PreProcessingPipelineError to be caught by the view
                    raise pre_err
                elif isinstance(pre_err, HttpResponse):
                    # This case should ideally not be hit if steps correctly return responses
                    # that get wrapped in PreProcessingPipelineError above. Log a warning.
                    logger.warning(
                        f"Pre-processing step {step_index} raised an unwrapped HttpResponse directly. Wrapping it now.")
                    # Attempt to save log, similar to the explicit check
                    self.request_log.status_code = pre_err.status_code
                    try:
                        self.request_log.save()
                        logger.debug(
                            f"Saved Request log entry ID {self.request_log.id} (unwrapped HttpResponse short-circuit)")
                    except Exception as save_err:
                        logger.error(
                            f"Failed to save Request log during unwrapped HttpResponse short-circuit: {save_err}",
                            exc_info=True)
                    raise PreProcessingPipelineError(response=pre_err, step_index=step_index)
                else:
                    # Handle other exceptions (BackendProcessingError or unexpected)
                    logger.error(f"Error during pre-processing step {step_index} for path '{relative_path}': {pre_err}",
                                 exc_info=True)
                    self.request_log.status_code = 500
                    try:
                        self.request_log.save()
                    except Exception as save_err:
                        logger.error(f"Failed to save Request log during pre-processing error handling: {save_err}",
                                     exc_info=True)
                    # Raise custom exception, chaining the original error
                    error_response = JsonResponse({'error': 'Internal gateway error during request pre-processing'},
                                                  status=500)
                    raise PreProcessingPipelineError(response=error_response, step_index=step_index)

        logger.debug(f"Pre-processing pipeline completed successfully for path '{relative_path}'.")
