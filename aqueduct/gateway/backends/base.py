import abc
import re
from typing import Optional, Callable, Union
import logging

from django.http import HttpRequest, HttpResponse, Http404
import requests
from requests import Request as RequestsRequest

from management.models import Endpoint, Model, Usage, Request

logger = logging.getLogger(__name__)


class AIGatewayBackend(abc.ABC):
    """
    Abstract base class for backend-specific logic within the AI Gateway.
    Subclasses handle tasks like finding the correct model and extracting usage
    based on the specifics of the target API (e.g., OpenAI, Anthropic).

    Instances are initialized with the Django HttpRequest and the resolved Endpoint.
    The associated Model instance (or None) is resolved during initialization
    and stored in self.model.
    """
    def __init__(self, request: HttpRequest, endpoint: Endpoint):
        """
        Initializes the backend instance and resolves the associated Model.

        Args:
            request (HttpRequest): The incoming Django request.
            endpoint (Endpoint): The resolved target Endpoint instance.
        """
        if not request or not endpoint:
            # Should not happen if instantiated correctly in the view
            raise ValueError("AIGatewayBackend requires a valid HttpRequest and Endpoint for initialization.")
        self.request = request
        self.endpoint = endpoint
        self.model: Optional[Model] = self._resolve_model() # Resolve and store model
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
