from pathlib import Path
from typing import ClassVar

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from management.models import Request, Usage


class RequestUsageTest(SimpleTestCase):
    def test_reasoning_tokens_are_bounded_by_output_tokens(self):
        for reported, expected in [(-5, 0), (50, 20), (12, 12)]:
            with self.subTest(reported=reported):
                req = Request()
                req.token_usage = Usage(
                    input_tokens=10, output_tokens=20, reasoning_tokens=reported
                )
                self.assertEqual(req.reasoning_tokens, expected)
                self.assertEqual(req.token_usage.reasoning_tokens, expected)
                self.assertEqual(req.token_usage.total_tokens, 30)

    def test_usage_arithmetic_preserves_token_subsets(self):
        first = Usage(
            input_tokens=100, output_tokens=20, cached_input_tokens=80, reasoning_tokens=12
        )
        second = Usage(
            input_tokens=50, output_tokens=10, cached_input_tokens=30, reasoning_tokens=6
        )
        combined = first + second
        self.assertEqual(
            combined,
            Usage(input_tokens=150, output_tokens=30, cached_input_tokens=110, reasoning_tokens=18),
        )
        self.assertEqual(combined.total_tokens, 180)
        self.assertEqual(combined - second, first)


@override_settings(
    LITELLM_ROUTER_CONFIG_FILE_PATH=(
        Path(__file__).resolve().parents[3] / "example_router_config.yaml"
    ),
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    },
)
class UsageDashboardTokenDetailsTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self):
        self.client.force_login(get_user_model().objects.get(pk=1))

    def test_dashboard_shows_subsets_without_double_counting(self):
        Request.objects.bulk_create(
            [
                Request(
                    token_id=1,
                    model="gpt-4.1-nano",
                    input_tokens=100,
                    cached_input_tokens=80,
                    output_tokens=20,
                    reasoning_tokens=12,
                    status_code=200,
                ),
                Request(
                    token_id=1,
                    model="gpt-4.1-nano",
                    input_tokens=50,
                    cached_input_tokens=30,
                    output_tokens=10,
                    reasoning_tokens=6,
                    status_code=200,
                ),
            ]
        )
        response = self.client.get(reverse("usage"))
        self.assertEqual(response.context["input_tokens"], 150)
        self.assertEqual(response.context["output_tokens"], 30)
        self.assertEqual(response.context["total_tokens"], 180)
        self.assertContains(response, "Of which 110 were cached")
        self.assertContains(response, "Of which 18 were reasoning tokens")

    def test_empty_dashboard_shows_zero_reasoning_tokens(self):
        response = self.client.get(reverse("usage"))
        self.assertEqual(response.context["reasoning_tokens"], 0)
        self.assertEqual(response.context["total_tokens"], 0)
        self.assertContains(response, "Of which 0 were reasoning tokens")
