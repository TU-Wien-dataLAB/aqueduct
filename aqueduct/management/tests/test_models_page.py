from typing import ClassVar
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse


@override_settings(
    STORAGES={
        "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
)
class ModelsPageTest(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def test_displays_configured_model_metadata(self):
        self.client.force_login(get_user_model().objects.get(pk=1))
        config = {
            "model_list": [
                {
                    "model_name": "example",
                    "model_info": {
                        "contextWindow": 131072,
                        "maxTokens": 8192,
                        "thinkingLevels": ["low", "high", "max"],
                    },
                }
            ]
        }
        with patch("management.views.models.get_router_config", return_value=config):
            response = self.client.get(reverse("models"))
        self.assertContains(response, "Context window")
        self.assertContains(response, "131,072")
        self.assertContains(response, "Max output tokens")
        self.assertContains(response, "8,192")
        self.assertContains(response, "Thinking levels")
        for level in ("low", "high", "max"):
            self.assertContains(response, level)

    def test_existing_litellm_context_limit_and_missing_metadata(self):
        self.client.force_login(get_user_model().objects.get(pk=1))
        config = {
            "model_list": [
                {"model_name": "legacy", "model_info": {"max_tokens": 32768}},
                {"model_name": "no-metadata"},
            ]
        }
        with patch("management.views.models.get_router_config", return_value=config):
            response = self.client.get(reverse("models"))
        self.assertContains(response, "Context window", count=1)
        self.assertContains(response, "32,768")
        self.assertNotContains(response, "Max output tokens")
        self.assertNotContains(response, "Thinking levels")

    def test_model_card_link_requires_http_url(self):
        self.client.force_login(get_user_model().objects.get(pk=1))
        config = {
            "model_list": [
                {
                    "model_name": "safe",
                    "model_info": {"model_card_url": "https://huggingface.co/example/model"},
                },
                {"model_name": "unsafe", "model_info": {"model_card_url": "javascript:alert(1)"}},
            ]
        }
        with patch("management.views.models.get_router_config", return_value=config):
            response = self.client.get(reverse("models"))
        self.assertContains(response, 'href="https://huggingface.co/example/model"')
        self.assertContains(response, "Model card", count=1)
        self.assertNotContains(response, 'href="javascript:')
