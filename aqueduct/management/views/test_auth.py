import logging
import uuid
from datetime import timedelta
from http import HTTPStatus
from typing import Any

from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.handlers.asgi import ASGIRequest
from django.db import transaction
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST

from gateway.views.decorators import token_authenticated
from gateway.views.utils import RawJsonResponse
from management.models import Org, Token, UserProfile

User = get_user_model()

log = logging.getLogger("aqueduct")

TEST_TOKEN_VALIDITY_HRS = 2


def _create_db_token() -> tuple[str, str]:
    """A synchronous wrapper for the database operations.

    Because `transaction.atomic()` does not work in async context."""
    user_id = uuid.uuid4().hex[:8]

    with transaction.atomic():
        user = User.objects.create_user(
            username=f"loadtest-{user_id}", email=f"loadtest-{user_id}@test.com"
        )
        user.groups.add(Group.objects.get(name="user"))
        org, _ = Org.objects.get_or_create(name="Performance Test")
        _ = UserProfile.objects.create(user=user, org=org, group="user")
        token = Token(
            user=user,
            name=f"loadtest-{user_id}",
            expires_at=timezone.now() + timedelta(hours=TEST_TOKEN_VALIDITY_HRS),
        )
        token_value = token._set_new_key()
        token.save()

    return token_value, user.username


@csrf_exempt
@require_POST
@token_authenticated(token_auth_only=True)
async def generate_test_token(request: ASGIRequest, *args: Any, **kwargs: Any) -> RawJsonResponse:
    """
    Generate a new `User` + `UserProfile` + `Token` for load testing.

    Token's validity timespan is limited.
    Only enabled when LOAD_TESTING=True to prevent abuse in production.
    """
    if not getattr(settings, "LOAD_TESTING", False):
        return RawJsonResponse(
            {"error": "Test token generation only available in load-testing mode"},
            status=HTTPStatus.FORBIDDEN,
        )

    try:
        token_value, username = await sync_to_async(_create_db_token)()
        return RawJsonResponse(
            {"token": token_value, "username": username}, status=HTTPStatus.CREATED
        )
    except Exception as e:
        log.exception("Failed to generate test token for load testing")
        return RawJsonResponse({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


@csrf_exempt
@require_http_methods(["DELETE"])
@token_authenticated(token_auth_only=True)
async def cleanup_test_token(
    request: ASGIRequest, token: Token, *args: Any, **kwargs: Any
) -> RawJsonResponse:
    """
    Delete the test `User` and its related objects (particularly: `Token`) after load testing.

    Token used for authenticating this request is the same that will be deleted.
    Only enabled when LOAD_TESTING=True to prevent abuse in production.
    """
    if not getattr(settings, "LOAD_TESTING", False):
        return RawJsonResponse(
            {"error": "Test token generation only available in load-testing mode"},
            status=HTTPStatus.FORBIDDEN,
        )

    username = token.user.username
    if not username.startswith("loadtest-"):
        log.warning("Attempted to delete non-loadtest user via cleanup endpoint: %s", username)
        return RawJsonResponse(
            {"error": "Only loadtest users can be deleted via this endpoint"},
            status=HTTPStatus.FORBIDDEN,
        )

    # Delete User; cascades to Token, UserProfile via FK
    await token.user.adelete()

    log.info("Cleaned up load test user: %s", username)
    return RawJsonResponse({"deleted": True, "username": username})
