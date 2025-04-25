import json
import logging
from typing import Optional, Callable, Union

from django.http import HttpRequest, Http404, HttpResponse, JsonResponse
import openai as openai_sdk
import requests

from gateway.backends.base import AIGatewayBackend
from management.models import Endpoint, Model, Usage, Request

logger = logging.getLogger(__name__)


def _transform_models(backend: 'AIGatewayBackend', request: HttpRequest, request_log: Request,
                      response: HttpResponse) -> HttpResponse:
    """
    Transforms the /models list response from an OpenAI-compatible API.
    - Filters the models to include only those defined for the request's endpoint (self.endpoint).
    - Replaces the 'id' of each model with its 'display_name' from the database.
    """
    try:
        # 1. Parse the original response content
        # Assuming the response content is valid JSON for OpenAI's SyncPage[Model]
        response_models = openai_sdk.pagination.SyncPage[openai_sdk.types.Model].model_validate_json(
            response.content)

        # 2. Get the Aqueduct Endpoint associated with this request (use self.endpoint)
        endpoint = backend.endpoint
        if not endpoint:
            # Should be unlikely due to __init__ validation
            logger.error(
                f"Post-processing: Endpoint not found on backend instance for model transformation (Request log ID: {request_log.id}). Returning original response.")
            return response

        # 3. Fetch allowed Models from the database for this Endpoint
        db_models = Model.objects.filter(endpoint=endpoint)

        # 4. Create a mapping from the internal model name (OpenAI ID) to the desired display name
        name_to_display_name = {model.name: model.display_name for model in db_models}
        logger.debug(f"Transforming models for endpoint '{endpoint.slug}'. DB models map: {name_to_display_name}")

        # 5. Filter and transform the models from the OpenAI response
        transformed_data = []
        original_model_count = len(response_models.data)
        for openai_model in response_models.data:
            # Check if the model's ID (e.g., 'gpt-4') is defined in our DB for this endpoint
            if openai_model.id in name_to_display_name:
                # Replace the ID with the display name from our DB
                display_name = name_to_display_name[openai_model.id]
                original_id = openai_model.id
                openai_model.id = display_name  # Modify the Pydantic model instance
                transformed_data.append(openai_model)
                logger.debug(
                    f"Kept and transformed model ID '{original_id}' to '{display_name}' for endpoint '{endpoint.slug}'.")
            else:
                # Filter out models not defined for this endpoint in our database
                logger.debug(
                    f"Filtered out model ID '{openai_model.id}' as it's not defined for endpoint '{endpoint.slug}'.")

        # 6. Update the data list in the SyncPage object
        response_models.data = transformed_data

        # 7. Serialize the modified SyncPage object back to JSON
        transformed_content = response_models.model_dump_json()

        # 8. Update the existing HttpResponse with the transformed content
        response.content = transformed_content
        response['Content-Type'] = 'application/json'
        # Django might automatically update Content-Length, but removing it is safer
        if 'Content-Length' in response:
            del response['Content-Length']

        logger.info(
            f"Transformed model list for endpoint '{endpoint.slug}'. Original count: {original_model_count}, New count: {len(transformed_data)}.")
        return response

    except json.JSONDecodeError:
        endpoint_slug_for_log = endpoint.slug if endpoint else "<unknown>"
        logger.error(
            f"Failed to decode JSON from OpenAI models response for endpoint '{endpoint_slug_for_log}'.",
            exc_info=True)
        # Return original response if parsing fails
        return response
    except Exception as e:
        endpoint_slug_for_log = endpoint.slug if endpoint else "<unknown>"
        logger.error(
            f"Unexpected error transforming OpenAI models response for endpoint '{endpoint_slug_for_log}': {e}",
            exc_info=True)
        # Return original response in case of any other error
        return response


def _validate_and_transform_model_in_request(backend: 'AIGatewayBackend', django_request: HttpRequest,
                                             request_log: Request, relay_request: requests.Request,
                                             response: Optional[HttpResponse]) -> Union[
    requests.Request, HttpResponse]:
    """
    Pre-processing step to validate the requested model against the endpoint's
    allowed models (using display_name) and transform it to the internal name
    within the `requests.Request` object.

    - Parses the request body (`relay_request.data`) to find the 'model' field.
    - Checks if the provided model name exists as a `display_name` for any `Model`
      associated with the `endpoint`.
    - If found, replaces the `display_name` in `relay_request.data` with the
      corresponding `model.name` (internal name) and updates the Content-Length header
      in `relay_request.headers`.
    - If not found, returns a 404 JsonResponse.
    - If the body is missing, not JSON, or missing the 'model' key, returns the
      unmodified `relay_request`.

    Args:
        backend: The OpenAIBackend instance.
        django_request: The original incoming Django HttpRequest (unused).
        request_log: The Request log model instance.
        relay_request: The `requests.Request` object being prepared. This object
                        may be modified in place (specifically `.data` and `.headers`).
        response: The final Django response object (None for pre-processing).

    Returns:
        Union[requests.Request, HttpResponse]: The (potentially modified) `relay_request` on success,
                                               or a JsonResponse on validation failure.
    """
    request_body = relay_request.data
    if not request_body:
        logger.debug("Pre-processing: Request body is empty, skipping model validation/transformation.")
        return relay_request  # Return unmodified object

    try:
        data = json.loads(request_body)
    except json.JSONDecodeError:
        logger.warning("Pre-processing: Failed to decode JSON from request body, skipping model validation.")
        return relay_request  # Let the target API handle malformed JSON, return unmodified

    requested_model_display_name = data.get('model')
    if not requested_model_display_name:
        logger.debug("Pre-processing: Request JSON missing 'model' key, skipping validation/transformation.")
        return relay_request  # No model specified, return unmodified

    try:
        # Use self.endpoint directly
        endpoint = backend.endpoint

        model_instance = Model.objects.get(
            display_name=requested_model_display_name,
            endpoint=endpoint  # Use self.endpoint
        )
        internal_model_name = model_instance.name

        if requested_model_display_name != internal_model_name:
            data['model'] = internal_model_name
            new_body_bytes = json.dumps(data).encode('utf-8')

            # Update the request object's data and headers
            relay_request.data = new_body_bytes
            relay_request.headers['Content-Length'] = str(len(new_body_bytes))

            logger.info(
                f"Pre-processing: Transformed model '{requested_model_display_name}' to '{internal_model_name}' for endpoint '{endpoint.slug}'.")
            return relay_request  # Return modified object
        else:
            # Model provided was already the internal name (and it's allowed)
            logger.debug(
                f"Pre-processing: Model '{internal_model_name}' provided directly and is valid for endpoint '{endpoint.slug}'.")
            return relay_request  # Return unmodified object

    except Model.DoesNotExist:
        logger.warning(
            f"Pre-processing: Model with display_name '{requested_model_display_name}' not found or not allowed for endpoint '{endpoint.slug}'.")
        return JsonResponse(
            {'error': f"Model '{requested_model_display_name}' is not available for this endpoint."},
            status=404
        )
    except Exception as e:
        # Capture the correct endpoint slug for logging
        endpoint_slug_for_log = endpoint.slug if endpoint else "<unknown>"
        logger.error(
            f"Pre-processing: Unexpected error validating/transforming model for endpoint '{endpoint_slug_for_log}': {e}",
            exc_info=True)
        return JsonResponse({"error": "Internal gateway error during model validation"}, status=500)


class OpenAIBackend(AIGatewayBackend):
    """
    Backend implementation for OpenAI-compatible API endpoints.
    """

    def __init__(self, request: HttpRequest, endpoint: Endpoint):
        super().__init__(request, endpoint)
        logger.debug(f"OpenAIBackend initialized for endpoint '{self.endpoint.slug}'")

    def get_model(self) -> Optional[Model]:
        """
        Extracts the model name from the request body (JSON 'model' key)
        and retrieves the Model object belonging to the specified endpoint (self.endpoint).
        Returns None if 'model' key is missing, JSON is invalid, or model not found.

        NOTE: This uses the *internal* model name (e.g., 'gpt-4'). The pre-processing
        step `_validate_and_transform_model_in_request` handles mapping from display names.
        """
        model_name = None  # Initialize model_name
        try:
            if not self.request.body:
                return None  # Return None if body is empty

            data = json.loads(self.request.body)
            model_name = data.get('model')

            if not model_name:
                return None  # Return None if 'model' key is missing

            # Ensure the model belongs to the correct endpoint (self.endpoint)
            model = Model.objects.get(display_name=model_name, endpoint=self.endpoint)
            return model

        except json.JSONDecodeError:
            return None  # Return None if JSON is invalid
        except Model.DoesNotExist:
            # Use model_name captured earlier in the log message
            log_model_name = model_name if model_name else "<not provided>"
            logger.warning(
                f"Model with internal name '{log_model_name}' not found for endpoint '{self.endpoint.slug}'.")
            return None  # Return None if model not found
        except Exception as e:  # Catch only truly unexpected errors
            logger.error(f"Unexpected error getting model for endpoint '{self.endpoint.slug}': {e}", exc_info=True)
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
                            # Find the closing brace `}}` for the usage object
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

    def pre_processing_endpoints(self) -> dict[str, list[Callable[
        ['AIGatewayBackend', HttpRequest, 'Request', requests.Request, Optional[HttpResponse]], Union[requests.Request, HttpResponse]]]]:
        """
        Returns a dictionary of path patterns to pre-processing callables for OpenAI.
        """
        return {
            # Match paths commonly requiring a 'model' parameter in the body
            r"^(v1/)?(chat/completions|completions|embeddings)$": [
                _validate_and_transform_model_in_request
            ],
        }

    def post_processing_endpoints(self) -> dict[
        str, list[Callable[['AIGatewayBackend', HttpRequest, 'Request', HttpResponse], HttpResponse]]]:
        """
        Returns a dictionary of path patterns to post-processing callables for OpenAI.
        """
        return {
            # Match 'models' or 'v1/models' exactly
            r"^(v1/)?models$": [_transform_models],
        }
