import abc
import asyncio
import dataclasses
import json
import re
from typing import Optional, Callable, Union, Tuple, Coroutine, Any, Mapping
import logging
import time

import httpx  # Replace requests with httpx
from asgiref.sync import async_to_sync, sync_to_async
from django.http import HttpRequest, HttpResponse, Http404, HttpResponseServerError, JsonResponse, StreamingHttpResponse
from django.utils import timezone

from management.models import Endpoint, Model, Usage, Request, Token

logger = logging.getLogger(__name__)


# --- Custom Exceptions ---

class PreProcessingPipelineError(Exception):
    def __init__(self, response: HttpResponse, step_index: Optional[int] = None):
        self.response = response
        self.step_index = step_index
        super().__init__(str(response.content))


@dataclasses.dataclass
class MutableRequest:
    method: str
    url: str
    headers: dict | None = None,
    json: dict[str, Any] | None = None
    timeout: float | None = 60

    def build(self, client: httpx.AsyncClient) -> httpx.Request:
        new_body_bytes = json.dumps(self.json).encode('utf-8')

        # Create a new httpx.Request with updated content and headers
        # Keep original method, url, and other headers (modify Content-Length)
        new_headers = httpx.Headers(self.headers)
        new_headers['Content-Length'] = str(len(new_body_bytes))

        return client.build_request(
            method=self.method,
            url=self.url,
            headers=self.headers,
            json=self.json,
            timeout=self.timeout,
        )


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

    model: Optional[Model]
    token: Token
    request_log: Request
    relay_request: httpx.Request

    def __init__(self, request: HttpRequest, endpoint: Endpoint, async_client: httpx.AsyncClient):
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
        self.async_client = async_client

    async def initialize(self):
        self.model: Optional[Model] = await self._resolve_model()  # Resolve and store model
        self.token: Token = await self._resolve_token()  # Resolve and store token (raises Http404 if not found)
        self.request_log: Request = self._create_request_log()  # Create initial log entry
        self.relay_request: MutableRequest = self._prepare_relay_request()  # Prepare the outbound request

        # Run pre-processing immediately after preparing the request
        await self._run_pre_processing_pipeline()

    @abc.abstractmethod
    async def _resolve_model(self) -> Optional[Model]:
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

    async def _resolve_token(self) -> Token:
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
                    token = await sync_to_async(Token.find_by_key)(token_key)
            except IndexError:
                logger.warning("Could not parse Bearer token from Authorization header.")
            except Exception as e:
                logger.error(f"Error looking up token from header: {e}", exc_info=True)

        # Fallback: If no token from header and user is authenticated
        user = getattr(self.request, 'user', None)  # Safely get user
        if token is None and user and user.is_authenticated:
            logger.debug(f"No valid Bearer token in header, attempting to find token for user {user.email}")
            # Fetch the first available token for this user.
            token = await user.custom_auth_tokens.order_by('pk').afirst()

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

    def _prepare_relay_request(self) -> MutableRequest:
        """
        Prepares the initial MutableRequest object for relaying.
        This includes setting the URL, method, headers, and body.
        """
        remaining_path = self.request.resolver_match.kwargs.get('remaining_path', '')
        # Ensure leading/trailing slashes are handled correctly for joining
        target_api_base = self.endpoint.url.rstrip('/')
        remaining_path_cleaned = remaining_path.lstrip('/')
        target_url = f"{target_api_base}/{remaining_path_cleaned}"

        # Prepare initial headers and body for the outbound request
        # Copy relevant headers, being careful about case and sensitive info
        outbound_headers = {
            "Content-Type": self.request.content_type or "application/json",
            # Example: "Accept": request.headers.get("Accept", "*/*"),
        }
        # Add/replace auth header for the target endpoint
        self.add_auth_header(outbound_headers)  # Modifies headers in place

        if self.is_streaming_request():
            outbound_headers["Accept"] = "text/event-stream"

        # Get body from original request
        outbound_body = self.request.body
        try:
            json_content = json.loads(outbound_body)
        except json.decoder.JSONDecodeError:
            json_content = None

        # Create the MutableRequest object
        request_object = MutableRequest(
            method=self.request.method,
            url=target_url,
            headers=outbound_headers,
            json=json_content,
        )

        logger.debug(f"Prepared initial relay request: {request_object.method} {request_object.url}")
        return request_object

    async def _send_relay_request(self, timeout: int) -> Tuple[HttpResponse, Optional[int]]:
        """
        Sends the prepared relay request using httpx, handles exceptions,
        and creates the initial Django HttpResponse.

        Args:
            timeout (int): Timeout in seconds for the request.

        Returns:
            Tuple containing:
            - HttpResponse: The initial Django HttpResponse (could be success, relayed error, or gateway error like 502/504).
            - int | None: The response time in milliseconds, or None if timing failed.
        """
        start_time = time.monotonic()
        target_url = str(self.relay_request.url)  # httpx.URL needs conversion to str
        response: HttpResponse  # Will hold the final Django response

        try:
            # Use httpx.Client for synchronous requests
            # Send the prepared httpx.Request object
            relayed_response = await self.async_client.send(self.relay_request.build(self.async_client))
            relayed_response.read()

            # Relay successful (even if target returned 4xx/5xx)
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            logger.debug(
                f"Relay to {target_url} completed: Status {relayed_response.status_code}, Time {response_time_ms}ms")
            # Create Django response from successful relay
            response = self._create_django_response(relayed_response)

        except httpx.TimeoutException:
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            logger.warning(f"Gateway timeout after {response_time_ms}ms relaying request to {target_url}")
            response = JsonResponse({"error": "Gateway timeout"}, status=504)

        except httpx.ConnectError as e:  # More specific error for connection issues
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            logger.error(
                f"Connection error relaying request to {target_url} (No response received), Time: {response_time_ms}ms: {e}",
                exc_info=False
            )
            response = JsonResponse({"error": "Gateway connection error"}, status=502)

        except httpx.RequestError as e:  # Catch other httpx request-related errors
            end_time = time.monotonic()
            response_time_ms = int((end_time - start_time) * 1000)
            # Check if the error has a response attached (e.g., from follow_redirects errors)
            if hasattr(e, 'response') and e.response:
                logger.error(
                    f"Error relaying request to {target_url} (Target Status: {e.response.status_code}), Time: {response_time_ms}ms: {e}",
                    exc_info=False
                )
                # Use the response from the exception if available
                response = self._create_django_response(e.response)
            else:
                logger.error(
                    f"Generic HTTP error relaying request to {target_url}, Time: {response_time_ms}ms: {e}",
                    exc_info=False  # Usually don't need full stack trace for request errors
                )
                # Default to 502 Bad Gateway if no response is available
                response = JsonResponse({"error": "Gateway request error"}, status=502)

        except Exception as e:  # Catch any other unexpected errors during send
            end_time = time.monotonic()
            response_time_ms = int((time.monotonic() - start_time) * 1000) if start_time else None
            logger.error(f"Unexpected error during request relay attempt to {target_url}: {e}", exc_info=True)
            # Unexpected error, create a 500 response
            response = JsonResponse({"error": "Internal gateway error during relay attempt"}, status=500)

        # Ensure response_time_ms is set if an error occurred very early
        if response_time_ms is None and start_time:
            response_time_ms = int((time.monotonic() - start_time) * 1000)

        return response, response_time_ms

    def _create_django_response(self, relayed_response: httpx.Response) -> HttpResponse:
        """
        Creates a Django HttpResponse from an httpx.Response object.

        Args:
            relayed_response (httpx.Response): The successful response from the relay.
                                                Assumes .read() has been called.

        Returns:
            HttpResponse: The corresponding Django HttpResponse.
        """
        # Content should be already read into relayed_response.content
        response_content = relayed_response.content

        # Construct Django HttpResponse
        django_response = HttpResponse(
            content=response_content,
            status=relayed_response.status_code,
            content_type=relayed_response.headers.get('Content-Type')
        )

        # Copy relevant headers from relayed response to Django response
        # httpx handles hop-by-hop headers filtering implicitly in many cases,
        # but we still apply the standard list for robustness.
        hop_by_hop_headers = {  # Use a set for faster lookups
            'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
            'te', 'trailers', 'transfer-encoding', 'upgrade',
            'content-encoding'  # Often handled by transport but exclude explicitly
        }
        for key, value in relayed_response.headers.items():
            if key.lower() not in hop_by_hop_headers:
                django_response[key] = value

        return django_response

    async def request_non_streaming(self, timeout: int = 60) -> HttpResponse:
        """
        Synchronously sends the prepared relay request, processes the response,
        runs post-processing, and logs the outcome.

        Args:
            timeout (int): Timeout in seconds for the request. Default 60.

        Returns:
            HttpResponse | JsonResponse: The final Django response object.
        """
        final_response: Optional[HttpResponse] = None
        response_time_ms: Optional[int] = None
        target_url = self.relay_request.url  # Get URL before potential errors

        logger.info(
            f"Relaying {self.relay_request.method} request from {self.request.user.email} (Token: {self.token.name}) "
            f"to {target_url} via Endpoint '{self.endpoint.name}'"
        )

        try:
            # Step 1: Send relay request & get initial Django response (handles relay errors internally)
            initial_django_response, duration_ms = await self._send_relay_request(timeout)
            response_time_ms = duration_ms  # Store duration
            logger.debug(
                f"_send_relay_request completed. Initial Status: {initial_django_response.status_code}, Time: {response_time_ms}ms")

            # Step 2: Run post-processing pipeline
            # The pipeline internally checks status code before running steps
            final_response = await self._run_post_processing_pipeline(initial_django_response)
            logger.debug(f"Post-processing attempt complete. Final Status: {final_response.status_code}")

            # Step 3: Extract Usage after post-processing (if applicable)
            # Only attempt extraction if the model exists and the final response indicates success
            # and has content. Post-processing might have altered the response.
            if self.model and final_response.content and final_response.status_code <= 300:
                usage = self.extract_usage(final_response.content)
                self.request_log.token_usage = usage
        except Exception as e:
            # Catch unexpected errors *outside* the relay send itself (e.g., in post-processing or usage extraction)
            logger.error(f"Unexpected error during sync request processing (post-relay) for {target_url}: {e}",
                         exc_info=True)
            # Ensure response_time_ms is logged if available from relay attempt
            final_response = JsonResponse({"error": "Internal gateway error during processing"}, status=500)
        finally:
            # Final logging and saving, using the final_response determined above
            if final_response:
                self.request_log.status_code = final_response.status_code
            else:
                # Should not happen ideally, but set a default status if response is somehow None
                logger.error("Final response object was unexpectedly None in finally block.")
                self.request_log.status_code = 500  # Internal Server Error

            if response_time_ms is not None:
                self.request_log.response_time_ms = response_time_ms

            # Ensure usage is zero if not extracted (this might happen if post-processing fails)
            if self.request_log.input_tokens is None: self.request_log.input_tokens = 0
            if self.request_log.output_tokens is None: self.request_log.output_tokens = 0

            # Save the log entry if it hasn't been saved yet (e.g., by pre-processing short-circuit)
            if not self.request_log.pk:
                try:
                    await self.request_log.asave()
                    logger.debug(f"Saved Request log entry ID {self.request_log.id} in request_sync finally block")
                except Exception as final_save_e:
                    logger.error(f"Failed to save Request log in request_sync finally block: {final_save_e}",
                                 exc_info=True)
            else:
                logger.debug(f"Request log ID {self.request_log.id} was already saved (likely by pre-processing).")

        # Ensure we always return a response
        return final_response if final_response is not None else HttpResponseServerError("Gateway internal error")

    async def request_streaming(self, timeout: int = 60) -> StreamingHttpResponse:
        upstream_response = await self.async_client.send(request=self.relay_request.build(self.async_client), stream=True)

        async def stream():
            start_time = time.monotonic()
            chunks: bytes = b''
            upstream_response.raise_for_status()
            try:
                async for chunk in upstream_response.aiter_bytes():
                    chunks += chunk
                    yield chunk
            except asyncio.CancelledError:
                # Handle client disconnect
                raise
            finally:
                # Post-processing after stream finishes
                usage = self.extract_usage(chunks)
                self.request_log.token_usage = usage
                end_time = time.monotonic()
                self.request_log.response_time_ms = int((end_time - start_time) * 1000)
                await self.request_log.asave()

        response = StreamingHttpResponse(
            streaming_content=stream(),
            content_type=upstream_response.headers.get('Content-Type')
        )
        response.status_code = upstream_response.status_code
        self.request_log.status_code = upstream_response.status_code
        await self.request_log.asave()

        # # Copy useful headers if needed
        for header in ["Content-Disposition", "Content-Length"]:
            if header in upstream_response.headers:
                response[header] = upstream_response.headers[header]

        return response

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
    def is_streaming_request(self) -> bool:
        """
        Determines whether the current request is a streaming request.

        Returns:
            bool: True if the request is a streaming request, False otherwise.
        """
        pass

    @abc.abstractmethod
    def pre_processing_endpoints(self) -> dict[str, list[Callable[
        ['AIGatewayBackend'], Coroutine[Any, Any, Union[MutableRequest, HttpResponse]]]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, regex)
        to a *list* of async callable functions (coroutines) that perform pre-processing transformations
        on the `MutableRequest` object before it's sent.

        Each async callable in the list receives:
        - The backend instance (self)

        Each async callable must return either:
        - An `MutableRequest` object (potentially modified): Indicates success, processing continues with the returned object.
        - An HttpResponse: To short-circuit the process and return this response immediately (e.g., for validation errors).
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
        str, list[Callable[['AIGatewayBackend', HttpResponse], Coroutine[Any, Any, HttpResponse]]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, regex)
        to a *list* of async callable functions (coroutines) that perform the required post-processing transformations
        on the HttpResponse.

        Each async callable in the list receives:
        - The backend instance (self)
        - The original HttpRequest
        - The Request model instance (log entry)
        - The current HttpResponse (output of the previous step or initial response)

        Each async callable must return the (potentially modified) HttpResponse.
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

    async def _run_pre_processing_pipeline(self):
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
                result = await step_func(self)

                if isinstance(result, HttpResponse):
                    # A step wants to short-circuit with a specific response.
                    logger.warning(
                        f"Pre-processing pipeline returned an HttpResponse at step {step_index} "
                        f"(Status: {result.status_code}). Short-circuiting relay.")
                    # Update log status before raising
                    self.request_log.status_code = result.status_code
                    try:
                        await self.request_log.asave()
                        logger.debug(f"Saved Request log entry ID {self.request_log.id} (pre-processing short-circuit)")
                    except Exception as save_err:
                        logger.error(f"Failed to save Request log during pre-processing short-circuit: {save_err}",
                                     exc_info=True)
                    # Raise the specific exception wrapping the response
                    raise PreProcessingPipelineError(response=result, step_index=step_index)

                elif isinstance(result, MutableRequest):
                    # Success, update the backend's request instance for the next step or final relay
                    self.relay_request = result
                    logger.debug(f"Pre-processing step {step_index} completed successfully, relay_request updated.")
                else:
                    # Should not happen based on type hints, but handle defensively
                    logger.error(
                        f"Pre-processing step {step_index} for path '{relative_path}' returned an unexpected type: {type(result)}. Aborting.")
                    self.request_log.status_code = 500
                    try:
                        await self.request_log.asave()
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
                        await self.request_log.asave()
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
                        await self.request_log.asave()
                    except Exception as save_err:
                        logger.error(f"Failed to save Request log during pre-processing error handling: {save_err}",
                                     exc_info=True)
                    # Raise custom exception, chaining the original error
                    error_response = JsonResponse({'error': 'Internal gateway error during request pre-processing'},
                                                  status=500)
                    raise PreProcessingPipelineError(response=error_response, step_index=step_index)

        logger.debug(f"Pre-processing pipeline completed successfully for path '{relative_path}'.")

    # --- Post-Processing ---

    async def _run_post_processing_pipeline(self, response: HttpResponse) -> HttpResponse:
        """
        Executes the post-processing pipeline based on matching endpoint patterns.
        Modifies and returns the HttpResponse.

        Args:
            response (HttpResponse): The initial response from the relayed request.

        Returns:
            HttpResponse: The final response after post-processing.
        """
        if not self.requires_post_processing():
            logger.debug("Post-processing not required for this request path.")
            return response

        # Get path from the original Django request associated with this backend instance
        relative_path = self.request.resolver_match.kwargs.get('remaining_path', '').lstrip('/')

        # Only run pipeline if the initial response was potentially successful
        if response.status_code < 400:
            processing_pipeline = []
            matched_patterns = []
            # Find all matching pipelines and extend
            for pattern, pipeline in self.post_processing_endpoints().items():
                if re.fullmatch(pattern, relative_path):
                    processing_pipeline.extend(pipeline)
                    matched_patterns.append(pattern)

            if processing_pipeline:
                logger.info(
                    f"Running combined post-processing pipeline ({len(processing_pipeline)} steps) "
                    f"for path '{relative_path}' (matched patterns: {matched_patterns}) using {type(self).__name__}")
                # original_response_status = response.status_code # Keep original status for logging if needed
                for i, step_func in enumerate(processing_pipeline):
                    step_index = i + 1
                    try:
                        # Pass backend instance, request log, and current response
                        response = await step_func(self, response)
                        logger.debug(
                            f"Post-processing step {step_index} completed. Current status: {response.status_code}")
                        # Check for error status code (4xx or 5xx) introduced by the step
                        if response.status_code >= 400:
                            logger.warning(
                                f"Post-processing pipeline stopped early at step {step_index} "
                                f"(patterns {matched_patterns}) due to status code {response.status_code}.")
                            # Update log status before returning
                            self.request_log.status_code = response.status_code
                            # Error occurred, return the error response immediately
                            return response
                    except Exception as pp_err:
                        logger.error(
                            f"Error during post-processing step {step_index} for path '{relative_path}' "
                            f"(patterns {matched_patterns}): {pp_err}", exc_info=True)
                        # Update log status
                        self.request_log.status_code = 500
                        # Return a generic 500 error if a step fails unexpectedly
                        return JsonResponse(
                            {"error": f"Gateway error during response post-processing step {step_index}"},
                            status=500)

                logger.debug(
                    f"Combined post-processing pipeline completed successfully for path '{relative_path}' "
                    f"(patterns {matched_patterns}). Final status: {response.status_code}")
                # Update log status with final code from pipeline
                self.request_log.status_code = response.status_code
        else:
            logger.debug(
                f"Skipping post-processing for path '{relative_path}' (initial status code {response.status_code}), "
                f"because initial response was not successful.")
            # Log status remains the original unsuccessful one

        return response
