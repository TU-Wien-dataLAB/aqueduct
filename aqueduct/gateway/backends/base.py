import abc
from typing import Optional, Callable, Union

from django.http import HttpRequest, HttpResponse

from management.models import Endpoint, Model, Usage, Request


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
    def pre_processing_endpoints(self) -> dict[str, list[Callable[
        ['AIGatewayBackend', HttpRequest, 'Request', Optional[HttpResponse]], Union[HttpRequest, HttpResponse]]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, no regex)
        to a *list* of callable functions that perform pre-processing transformations.

        Each callable in the list receives:
        - The backend instance (self)
        - The current HttpRequest
        - The Request model instance (log entry)
        - An Optional[HttpResponse], always passed as None for pre-processing steps.

        Each callable must return either:
        - An HttpRequest: The (potentially modified) request to pass to the next step or to the relay.
        - An HttpResponse: To short-circuit the process and return this response immediately.
        """
        pass

    def requires_pre_processing(self, request: HttpRequest) -> bool:
        """
        Checks if the given request's remaining path matches any path defined
        as a key in the dictionary returned by pre_processing_endpoints.

        Args:
            request (HttpRequest): The incoming request.

        Returns:
            bool: True if the path matches a pattern requiring pre-processing, False otherwise.
        """
        remaining_path = request.resolver_match.kwargs.get('remaining_path', '')
        # Check if the cleaned path exists as a key in the pre-processing dict
        return remaining_path.lstrip('/') in self.pre_processing_endpoints()

    @abc.abstractmethod
    def post_processing_endpoints(self) -> dict[
        str, list[Callable[['AIGatewayBackend', HttpRequest, 'Request', HttpResponse], HttpResponse]]]:
        """
        Returns a dictionary mapping endpoint path patterns (relative strings, no regex)
        to a *list* of callable functions that perform the required post-processing transformations
        on the HttpResponse.

        Each callable in the list receives:
        - The backend instance (self)
        - The original HttpRequest
        - The Request model instance (log entry)
        - The current HttpResponse (output of the previous step or initial response)
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
