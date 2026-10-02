"""Tests the usage dashboard counts 499 (client-closed) as a failed request.

A 499 status code is >= 400, so the dashboard's ``failed_requests`` stat
(``status_code__gte=400``) must include it — confirming client-closed requests
show up in the dashboard's failed-requests count.
"""

from pathlib import Path

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings
from django.urls import reverse

from management.models import Org, Request, Token, UserGroup, UserProfile

User = get_user_model()

ROOT = Path(__file__).resolve().parents[3]

ALLOWED_MODEL = "gpt-4.1-nano"


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=str(ROOT / "example_router_config.yaml"))
class UsageDashboardFailedRequestTests(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="usage-org")
        self.user = User.objects.create_user(username="usageuser", email="usage@example.com")
        UserProfile.objects.create(user=self.user, org=self.org)
        Group.objects.get_or_create(name=UserGroup.USER.value)
        self.token = Token(name="usage-token", user=self.user)
        self.token._set_new_key()
        self.token.save()
        self.client.force_login(self.user)

    def _add_request(self, status_code: int) -> Request:
        return Request.objects.create(
            token=self.token,
            model=ALLOWED_MODEL,
            status_code=status_code,
            user_id=self.user.email,
            path="/chat/completions",
        )

    def test_499_counts_as_failed_request(self):
        self._add_request(status_code=499)

        resp = self.client.get(reverse("usage"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["failed_requests"], 1)
        self.assertEqual(resp.context["total_requests"], 1)

    def test_499_and_500_both_count_as_failed(self):
        self._add_request(status_code=200)
        self._add_request(status_code=499)
        self._add_request(status_code=500)

        resp = self.client.get(reverse("usage"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.context["total_requests"], 3)
        self.assertEqual(resp.context["failed_requests"], 2)
