from typing import ClassVar
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.tests.helpers import limit_grid

User = get_user_model()


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH)
class TokensPageTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

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
