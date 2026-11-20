from typing import ClassVar
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.rate_limiting import check_and_reserve
from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.models import LimitSet
from management.tests.helpers import limit_grid

User = get_user_model()


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class TokensPageTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

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
        limits = LimitSet.from_objects(profile, [profile.org])
        for _ in range(6):
            allowed, _ = check_and_reserve(limits, token_id, model=None)
            self.assertTrue(allowed)

        self.client.force_login(user)
        resp = self.client.get(reverse("token_detail", args=[1]))

        self.assertEqual(resp.status_code, 200)
        minute, hour, day = limit_grid(resp)["Requests"]
        self.assertEqual((minute["used"], minute["limit"], minute["percent"]), ("6", "of 10", "60"))
        self.assertEqual((hour["used"], hour["limit"], hour["percent"]), ("6", "of 600", "1"))
        self.assertEqual((day["used"], day["limit"], day["percent"]), ("6", "of 14,400", "0"))
        self.assertEqual({cell["meter"] for cell in (minute, hour, day)}, {"progress-primary"})

    def test_meters_warn_from_80_percent_and_turn_red_from_100(self):
        user = User.objects.get(pk=1)
        user.profile.requests_per_minute = 10
        user.profile.save(update_fields=["requests_per_minute"])
        none = {"in": 0, "out": 0}
        usage = {
            1: {
                "min": {"req": 0, **none},
                "hour": {"req": 540, **none},
                "day": {"req": 21600, **none},
            }
        }
        self.client.force_login(user)
        with mock.patch("management.limits.get_per_token_usage", return_value=usage):
            resp = self.client.get(reverse("token_detail", args=[1]))

        _minute, hour, day = limit_grid(resp)["Requests"]
        self.assertEqual(
            (hour["meter"], hour["percent"], hour["used"]), ("progress-warning", "90", "540")
        )
        self.assertEqual(
            (day["meter"], day["percent"], day["used"]), ("progress-error", "100", "21,600")
        )

    def test_page_says_no_limit_when_there_is_no_limit(self):
        user = User.objects.get(pk=1)
        self.client.force_login(user)
        resp = self.client.get(reverse("token_detail", args=[1]))

        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "No other limit applies to this key.")
        self.assertNotContains(resp, "None")
        self.assertNotContains(resp, "<progress")
