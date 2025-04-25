import abc
import re
from typing import Optional, Callable, Union

from django.http import HttpRequest, HttpResponse, Http404
import requests

from management.models import Endpoint, Model, Usage, Request


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
        ['AIGatewayBackend', HttpRequest, 'Request', requests.Request, Endpoint], Optional[HttpResponse]]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, regex)
        to a *list* of callable functions that perform pre-processing transformations
        on the `requests.Request` object before it's sent.

        Each callable in the list receives:
        - The backend instance (self)
        - The original Django HttpRequest
        - The Request model instance (log entry)
        - The `requests.Request` object prepared for the outbound call. The callable can modify this object's attributes (e.g., `.url`, `.headers`, `.data`) in place.
        - The target Endpoint instance.

        Each callable must return either:
        - None: Indicates success, processing continues.
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
        str, list[Callable[['AIGatewayBackend', HttpRequest, 'Request', HttpResponse], HttpResponse]]]:
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
