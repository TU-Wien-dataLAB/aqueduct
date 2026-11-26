import logging

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.handlers.asgi import ASGIRequest
from django.db import connection
from django.http import HttpResponse
from django.urls import reverse
from tos.middleware import UserAgreementMiddleware as TOSUserAgreementMiddleware

User = get_user_model()

log = logging.getLogger("aqueduct")


class UserAgreementMiddleware(TOSUserAgreementMiddleware):
    """
    Some middleware to check if users have agreed to the latest TOS
    """

    sync_capable = True
    async_capable = False

    def __call__(self, request) -> HttpResponse | None:
        response = self.process_request(request)
        if response is None:
            return self.get_response(request)
        return response

    def should_fast_skip(self, request: ASGIRequest) -> bool:
        if request.path_info.rstrip("/") == "/oidc/callback":
            return True

        if request.path_info.rstrip("/") == reverse("sso").rstrip("/"):
            return True

        if request.path_info.rstrip("/") == reverse("admin_sso").rstrip("/"):
            return True

        return super().should_fast_skip(request)


def _database_ready() -> bool:
    """Whether a database connection is available within `HEALTH_DB_TIMEOUT` and answers."""
    try:
        pool = getattr(connection, "pool", None)
        if pool is not None:
            # Take a connection from the pool directly, so the wait is bounded by
            # HEALTH_DB_TIMEOUT instead of POSTGRESQL_POOL_TIMEOUT. The pool is
            # opened lazily, and this may be the first database access.
            pool.open()
            with pool.connection(timeout=settings.HEALTH_DB_TIMEOUT) as conn:
                conn.execute("SELECT 1")
        else:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
            finally:
                connection.close()
    except Exception as e:
        log.warning("Readiness check failed, database unavailable: %r", e)
        return False
    return True


class HealthCheckMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: ASGIRequest) -> HttpResponse:
        path = request.path_info.rstrip("/")
        if path == "/health":
            return HttpResponse("ok", content_type="text/plain")

        if path == "/health/ready":
            if _database_ready():
                return HttpResponse("ok", content_type="text/plain")
            return HttpResponse("db unavailable", content_type="text/plain", status=503)

        return self.get_response(request)
