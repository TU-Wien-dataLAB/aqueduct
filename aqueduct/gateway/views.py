# gateway/views.py
from django.http import HttpResponse, JsonResponse, HttpResponseNotAllowed
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging

logger = logging.getLogger(__name__)


# --- Base Gateway View with Authentication ---

@method_decorator(csrf_exempt, name='dispatch')
class AIGatewayView(View):
    """
    Base class for AI Gateway views providing authentication.
    Handles authentication in the dispatch method.
    If authentication fails, returns an error response.
    If authentication succeeds, returns None, allowing subclass dispatch to continue.
    """
    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def _authenticate(self, request):
        # (Authentication logic remains the same as before)
        # ... (omitted for brevity, keep your previous _authenticate method here) ...
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logger.warning("Authentication failed: Missing or malformed Bearer token header.")
            return None

        try:
            token = auth_header.split(' ')[1]
            if not token:
                raise IndexError
        except IndexError:
            logger.warning("Authentication failed: Empty or badly formatted Bearer token.")
            return None

        is_valid = bool(token)  # Placeholder validation

        if is_valid:
            logger.info(f"Authentication successful for token starting with: {token[:4]}...")
            request.auth_info = {'token_prefix': token[:4], 'status': 'authenticated'}
            return request.auth_info
        else:
            logger.warning(f"Authentication failed: Invalid token provided (starts with {token[:4]}...).")
            return None

    def dispatch(self, request, *args, **kwargs):
        """
        Performs authentication.
        Returns:
            - JsonResponse with status 401 if authentication fails.
            - None if authentication succeeds, indicating the subclass should proceed.
        """
        auth_result = self._authenticate(request)

        if auth_result is None:
            # Return a 401 Unauthorized response if authentication fails
            return JsonResponse(
                {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided.'},
                status=401
            )

        # Authentication successful - return None to signal the subclass dispatch can run its logic.
        logger.debug("Base authentication successful. Allowing subclass dispatch.")
        return None


# --- V1 OpenAI Specific Gateway View ---

class V1OpenAIGateway(AIGatewayView):
    """
    Handles API requests prefixed with 'v1/'. Inherits authentication from AIGatewayView.
    Overrides dispatch to handle the request AFTER successful base class authentication.
    """

    def dispatch(self, request, *args, **kwargs):
        """
        Overrides the base dispatch. Calls base dispatch first for authentication.
        If authentication succeeds (base returns None), proceeds with V1-specific logic.
        """
        # Step 1: Call base class dispatch FOR AUTHENTICATION ONLY.
        auth_response = super().dispatch(request, *args, **kwargs)

        # Step 2: Check if the base class returned an authentication error response.
        if auth_response:
            # Authentication failed, return the error response immediately.
            return auth_response

        # Step 3: Authentication passed (base returned None). Proceed with V1 logic.
        remaining_path = kwargs.get('remaining_path', '')  # Will be '' if URL was exactly 'v1/'
        full_path = f"v1/{remaining_path}"
        auth_info = getattr(request, 'auth_info', {})  # Get auth info set by base class

        logger.info(
            f"V1OpenAIGateway dispatching {request.method} for {full_path} "
            f"(Auth status: {auth_info.get('status', 'unknown')})"
        )

        # --- Placeholder: V1 Request Handling Logic ---
        # All the logic that was in _process_request now goes here.
        # You can check request.method if behavior needs to differ.
        # Example:
        # if request.method == 'POST':
        #    # Handle POST specific logic (e.g., read body)
        #    pass
        # elif request.method == 'GET':
        #    # Handle GET specific logic
        #    pass
        #
        # For now, return a generic JSON response indicating success
        response_data = {
            "message": f"Successfully dispatched {request.method} via V1OpenAIGateway",
            "requested_path": full_path,
            "authenticated_as": auth_info.get('token_prefix', 'N/A') + '...'
            # Add actual processed data here in a real implementation
        }
        status_code = 200  # Default success code

        # --- End Placeholder ---

        return JsonResponse(response_data, status=status_code)

    # NOTE: Individual get(), post(), _process_request() methods are removed
    # as the logic is now consolidated within this overridden dispatch method.
