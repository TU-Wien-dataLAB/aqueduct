import abc
import re
from typing import Optional, Callable, Union
import logging

from django.http import HttpRequest, HttpResponse, Http404
from django.utils import timezone
import requests
from requests import Request as RequestsRequest

from management.models import Endpoint, Model, Usage, Request, Token

logger = logging.getLogger(__name__)


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
        and creates the initial Request log entry.

        Args:
            request (HttpRequest): The incoming Django request.
            endpoint (Endpoint): The resolved target Endpoint instance.

        Raises:
            Http404: If the Token cannot be resolved.
            ValueError: If request or endpoint is invalid.
        """
        if not request or not endpoint:
            # Should not happen if instantiated correctly in the view
            raise ValueError("AIGatewayBackend requires a valid HttpRequest and Endpoint for initialization.")
        self.request: HttpRequest = request
        self.endpoint: Endpoint = endpoint
        self.model: Optional[Model] = self._resolve_model() # Resolve and store model
        self.token: Token = self._resolve_token() # Resolve and store token (raises Http404 if not found)
        self.request_log: Request = self._create_request_log() # Create initial log entry
        self.relay_request: RequestsRequest = self._prepare_relay_request() # Prepare the outbound request

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
        user = getattr(self.request, 'user', None) # Safely get user
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
            model=self.model, # Use the resolved model from self
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
            "Content-Type": self.request.content_type or "application/json", # Use original content-type or default
            # Copy other relevant headers? Be careful not to leak internal info.
            # Example: "Accept": request.headers.get("Accept", "*/*"),
        }
        # Add/replace auth header for the target endpoint
        self.add_auth_header(outbound_headers) # Call the method now part of the backend

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
