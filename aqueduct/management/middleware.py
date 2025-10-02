from django.contrib.auth import get_user_model
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse
from django.urls import reverse

User = get_user_model()

from tos.middleware import UserAgreementMiddleware as TOSUserAgreementMiddleware


class UserAgreementMiddleware(TOSUserAgreementMiddleware):
    """
    Some middleware to check if users have agreed to the latest TOS
    """
    sync_capable = True
    async_capable = False

    def __call__(self, request):
        response = self.process_request(request)
        if response is None:
            return self.get_response(request)
        else:
            return response

    def should_fast_skip(self, request: ASGIRequest):
        if request.path_info.rstrip("/") == "/oidc/callback":
            return True

        if request.path_info.rstrip("/") == reverse("sso").rstrip("/"):
            return True

        if request.path_info.rstrip("/") == reverse("admin_sso").rstrip("/"):
            return True

        return super().should_fast_skip(request)


class HealthCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: ASGIRequest):
        if request.path_info.rstrip("/") == "/health":
            return HttpResponse("ok", content_type="text/plain")

        return self.get_response(request)
