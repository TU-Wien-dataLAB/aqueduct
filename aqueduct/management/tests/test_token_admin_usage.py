"""Tests for the Django admin Token changelist hourly/daily usage columns.

``TokenAdmin`` shows two extra ``list_display`` columns — "Hourly" and "Daily" —
rendering the per-token request-usage percentage against the derived hour/day
limits (``LimitSet.windows()``), sourced from the cache buckets via
``gateway.rate_limiting.get_per_token_usage``. Tokens without a request limit
render an em dash instead of a percentage.
"""

from typing import ClassVar

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.rate_limiting import check_and_reserve
from management.models import LimitSet, UserProfile

User = get_user_model()


# Render the admin changelist (which loads admin CSS via {% static %}) without the
# hashed-manifest staticfiles storage (no collectstatic is run in the test
# environment), so {% static %} resolves.
_RENDER_HTML = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)


@_RENDER_HTML
class TokenAdminUsageColumnsTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        cache.clear()
        self.admin = User.objects.create_superuser(
            username="admin", email="admin@example.com", password="admin"
        )

    def tearDown(self) -> None:
        cache.clear()

    def _set_per_minute_limit(self, profile: UserProfile, rpm: int) -> LimitSet:
        profile.requests_per_minute = rpm
        profile.save(update_fields=["requests_per_minute"])
        return LimitSet.from_objects(profile, profile.org)

    def test_changelist_shows_hourly_and_daily_usage_percentages(self):
        """Token 1 (user token) resolves limits via its UserProfile.

        With ``requests_per_minute=10`` the derived caps are 600/hour and
        14400/day (default multipliers 60/1440). Reserving 6 requests on the
        token then reads as 1% hourly and 0% daily.
        """
        profile = UserProfile.objects.get(pk=1)  # owner of Token 1 ("My Token")
        limits = self._set_per_minute_limit(profile, 10)
        for _ in range(6):
            allowed, _ = check_and_reserve(limits, token_id=1, model=None)
            self.assertTrue(allowed)

        self.client.force_login(self.admin)
        resp = self.client.get(reverse("admin:management_token_changelist"))

        self.assertEqual(resp.status_code, 200)
        # Column headers are rendered.
        self.assertContains(resp, "Hourly")
        self.assertContains(resp, "Daily")
        # Hourly: 6/600 = 1%, with the raw used/limit counts in the tooltip.
        self.assertContains(resp, 'title="6 / 600 requests"')
        self.assertContains(resp, ">1%</span>")
        # Daily: 6/14400 ~ 0%, with the raw counts in the tooltip.
        self.assertContains(resp, 'title="6 / 14400 requests"')
        self.assertContains(resp, ">0%</span>")

    def test_changelist_shows_em_dash_when_no_request_limit(self):
        """A token whose resolved limit has no per-minute cap renders "—"."""
        # Token 2 ("Initial token for TUna") is a service-account token; its
        # Team (Whale) has no per-minute limit in the fixture, so hour/day are
        # also None -> em dash.
        self.client.force_login(self.admin)
        resp = self.client.get(reverse("admin:management_token_changelist"))

        self.assertEqual(resp.status_code, 200)
        # No request limit -> the column renders the em-dash placeholder, and
        # no percentage is rendered for it.
        self.assertContains(resp, "—")
        self.assertNotContains(resp, "%</span>")
