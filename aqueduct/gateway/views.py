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
    Relies on Django's authentication middleware (configured with a custom backend)
    to populate request.user based on the Authorization header.
    If authentication fails, dispatch returns a 401 error response.
    """
    http_method_names = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options', 'trace']

    def dispatch(self, request, *args, **kwargs):
        """
        Checks if the user was successfully authenticated by the middleware.
        If not authenticated, returns a 401 Unauthorized response.
        If authenticated, calls the standard Django View dispatch mechanism,
        which will route the request to the appropriate HTTP method handler
        (e.g., get(), post()) on the subclass.
        """
        # Authentication middleware runs before dispatch and should set request.user
        if not request.user.is_authenticated:
            # User was not authenticated by any configured backend
            logger.warning("Authentication check failed in dispatch: request.user is not authenticated.")
            return JsonResponse(
                {'error': 'Authentication Required', 'detail': 'A valid Bearer token must be provided and valid.'},
                status=401
            )

        # Authentication successful - proceed with standard view dispatching
        logger.debug(f"User {request.user.email} authenticated. Proceeding with dispatch.")
        return super().dispatch(request, *args, **kwargs)


# --- V1 OpenAI Specific Gateway View ---

class V1OpenAIGateway(AIGatewayView):
    """
    Handles API requests prefixed with 'v1/'. Inherits authentication from AIGatewayView.
    Implements specific HTTP method handlers (e.g., post).
    """

    # --- Implement HTTP method handlers directly --- 
    # These will only be called if authentication succeeds in the base class dispatch

    def post(self, request, *args, **kwargs):
        """
        Handles POST requests to the /v1/... endpoint.
        Assumes authentication has already been handled by AIGatewayView.dispatch.
        """
        # request.user is guaranteed to be an authenticated User instance here
        remaining_path = kwargs.get('remaining_path', '')
        full_path = f"v1/{remaining_path}"

        logger.info(
            f"V1OpenAIGateway handling POST for {full_path} "
            f"(User: {request.user.email})"
        )

        # --- Placeholder: V1 POST Request Handling Logic --- 
        # Example: Process request.body, interact with services, etc.
        # Access user info via request.user
        # Access user profile via request.user.profile (if needed)

        # For now, return a generic JSON response indicating success
        response_data = {
            "message": f"Successfully handled POST via V1OpenAIGateway",
            "requested_path": full_path,
            "authenticated_user": request.user.email, # Use info from request.user
            "user_groups": [g.name for g in request.user.groups.all()] # Example: get user groups
            # Add actual processed data here in a real implementation
        }
        status_code = 200  # Default success code

        # --- End Placeholder ---

        return JsonResponse(response_data, status=status_code)

    # Add other methods (get, put, etc.) as needed
    def get(self, request, *args, **kwargs):
        # Example placeholder for GET
        remaining_path = kwargs.get('remaining_path', '')
        full_path = f"v1/{remaining_path}"
        logger.info(f"V1OpenAIGateway handling GET for {full_path} (User: {request.user.email})")
        return JsonResponse({"message": "GET requests not fully implemented yet.", "path": full_path}, status=501)

    # You might want to explicitly disallow methods you don't handle
    # by overriding http_method_not_allowed or simply not defining the methods.
    # The base http_method_names allows all common ones.
