import logging
import uuid
from datetime import timedelta
from http import HTTPStatus

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.db import transaction
from django.http import HttpRequest, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from gateway.views.decorators import token_authenticated
from management.models import Org, Token, UserProfile

User = get_user_model()

log = logging.getLogger("aqueduct")

TEST_TOKEN_VALIDITY_HRS = 2


@csrf_exempt
@require_POST
@token_authenticated
def generate_test_token(request: HttpRequest) -> JsonResponse:
    """
    Generate a new `User` + `UserProfile` + `Token` for load testing.

    Token's validity timespan is limited.
    Only enabled when LOAD_TESTING=True to prevent abuse in production.
    """
    if not getattr(settings, "LOAD_TESTING", False):
        return JsonResponse(
            {"error": "Test token generation only available in load-testing mode"},
            status=HTTPStatus.FORBIDDEN,
        )

    try:
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

        return JsonResponse(
            {"token": token_value, "username": user.username}, status=HTTPStatus.CREATED
        )
    except Exception as e:
        log.exception("Failed to generate test token for load testing")
        return JsonResponse({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


@csrf_exempt
@require_POST
@token_authenticated
def cleanup_test_token(request: HttpRequest) -> JsonResponse:
    """
    Delete the test `User` and its related objects (particularly: `Token`) after load testing.

    Token used for authenticating this request is the same that will be deleted.
    Only enabled when LOAD_TESTING=True to prevent abuse in production.
    """
    if not getattr(settings, "LOAD_TESTING", False):
        return JsonResponse(
            {"error": "Test token generation only available in load-testing mode"},
            status=HTTPStatus.FORBIDDEN,
        )

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JsonResponse(
            {"error": "Missing or invalid Authorization header"}, status=HTTPStatus.BAD_REQUEST
        )

    token_key = auth_header.split(" ", 1)[1]
    token = Token.find_by_key(token_key)

    if not token:
        return JsonResponse({"error": "Token not found"}, status=HTTPStatus.NOT_FOUND)

    username = token.user.username

    if not username.startswith("loadtest-"):
        log.warning("Attempted to delete non-loadtest user via cleanup endpoint: %s", username)
        return JsonResponse(
            {"error": "Only loadtest users can be deleted via this endpoint"},
            status=HTTPStatus.FORBIDDEN,
        )

    # Delete User; cascades to Token, UserProfile via FK
    token.user.delete()

    log.info("Cleaned up load test user: %s", username)
    return JsonResponse({"deleted": True, "username": username})
