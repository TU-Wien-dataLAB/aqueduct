"""Tests for the tokens list page — per-window request-usage progress bars.

Covers the ``_rate_usage.html`` include driven by
``gateway.rate_limiting.get_aggregate_usage`` on ``UserTokensView``, rendered
under each API key list item as three daisyUI linear ``progress`` bars
(per minute / per hour / per day).
"""

from typing import ClassVar

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.rate_limiting import check_and_reserve
from management.models import LimitSet

User = get_user_model()


# Render full management pages without the hashed-manifest staticfiles storage
# (no collectstatic is run in the test environment), so {% static %} resolves.
_RENDER_HTML = override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)


@_RENDER_HTML
class TokensPageTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        cache.clear()

    def tearDown(self) -> None:
        cache.clear()

    def test_page_renders_usage_progress_bars(self):
        user = User.objects.get(pk=1)
        profile = user.profile

        # Default multipliers (hour=60, day=1440) so the hour/day caps are
        # derived from the per-minute cap: 10/min -> 600/hour, 14400/day.
        # 6 reserved requests then read as min 60%, hour 1%, day ~0% — verifying
        # that the coarser windows scale independently of the per-minute cap.
        profile.requests_per_minute = 10
        profile.hourly_limit_multiplier = None  # -> settings default 60
        profile.daily_limit_multiplier = None  # -> settings default 1440
        profile.save(
            update_fields=[
                "requests_per_minute",
                "hourly_limit_multiplier",
                "daily_limit_multiplier",
            ]
        )

        token_id = 1  # "My Token" from the fixture
        limits = LimitSet.from_objects(profile, profile.org)
        for _ in range(6):
            allowed, _ = check_and_reserve(limits, token_id, model=None)
            self.assertTrue(allowed)

        self.client.force_login(user)
        resp = self.client.get(reverse("tokens"))

        self.assertEqual(resp.status_code, 200)
        # Two daisyUI linear progress bars are rendered (hour + day, side by side).
        self.assertTemplateUsed(resp, "management/_rate_usage.html")
        self.assertContains(resp, "progress-primary", count=2)
        # Bar fill values are clamped percentages of the derived caps.
        self.assertContains(resp, 'value="1"')  # hour: 6/600 = 1%
        self.assertContains(resp, 'value="0"')  # day:  6/14400 ~0%
        # Per-window used/limit text reflects the derived caps.
        self.assertContains(resp, "6/600")  # hour
        self.assertContains(resp, "6/14,400")  # day (intcomma)
        # The per-minute bar is not rendered.
        self.assertNotContains(resp, "6/10")

    def test_page_renders_em_dash_when_no_request_limit(self):
        """With no request limit configured, the bars render a faded "—" (no cap)."""
        user = User.objects.get(pk=1)
        self.client.force_login(user)
        resp = self.client.get(reverse("tokens"))

        self.assertEqual(resp.status_code, 200)
        # No request limit -> the include renders the em-dash placeholder, not a fill.
        self.assertContains(resp, "—")
        # No filled primary bar is rendered (every window has no request cap).
        self.assertNotContains(resp, "progress-primary")
