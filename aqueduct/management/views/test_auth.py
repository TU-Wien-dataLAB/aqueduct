import logging
import uuid
from datetime import timedelta
from http import HTTPStatus

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.core.handlers.asgi import ASGIRequest
from django.http import HttpResponseForbidden, JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from management.models import Org, Token, UserProfile

User = get_user_model()

log = logging.getLogger("aqueduct")


@csrf_exempt
@require_POST
def generate_test_token(request: ASGIRequest) -> JsonResponse | HttpResponseForbidden:
    """
    Generate a new user + profile + token for load testing.
    Only enabled when LOAD_TESTING=True to prevent abuse in production.
    """
    if not getattr(settings, "LOAD_TESTING", False):
        return HttpResponseForbidden("Test token generation only available in load-testing mode")

    try:
        user_id = uuid.uuid4().hex[:8]
        user = User.objects.create_user(
            username=f"loadtest-{user_id}", email=f"loadtest-{user_id}@test.com"
        )
        user.groups.add(Group.objects.get(name="user"))
        org, _ = Org.objects.get_or_create(name="E060")
        _ = UserProfile.objects.create(user=user, org=org, group="user")
        token = Token(
            user=user, name=f"locust-test-{user_id}", expires_at=timezone.now() + timedelta(hours=2)
        )
        token_value = token._set_new_key()
        token.save()

        return JsonResponse({"token": token_value, "user_id": user.id}, status=HTTPStatus.CREATED)
    except Exception as e:
        log.exception("Failed to generate test token for load testing")
        return JsonResponse({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
