import asyncio
from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponse, JsonResponse

from management.models import UserProfile

User = get_user_model()

from django import VERSION as DJANGO_VERSION
from django.conf import settings
from django.contrib.auth import SESSION_KEY as session_key
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.core.cache import caches
from django.http import HttpResponseRedirect
from django.urls import reverse_lazy
from django.utils.cache import add_never_cache_headers
from django.utils.deprecation import MiddlewareMixin

from tos.models import UserAgreement
from tos.middleware import UserAgreementMiddleware as TOSUserAgreementMiddleware


class UserAgreementMiddleware(TOSUserAgreementMiddleware):
    """
    Some middleware to check if users have agreed to the latest TOS
    """
    sync_capable = True
    async_capable = False

    def __call__(self, request):
        if request.path_info.rstrip("/") == "/oidc/callback":
            return self.get_response(request)


        response = self.process_request(request)
        if response is None:
            return self.get_response(request)
        else:
            return response


class HealthCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: ASGIRequest):
        if request.path_info.rstrip("/") == "/health":
            return HttpResponse("ok", content_type="text/plain")

        return self.get_response(request)
