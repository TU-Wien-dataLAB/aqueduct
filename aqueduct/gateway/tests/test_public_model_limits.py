"""Public-model budgets through gateway endpoints; no distributed-state claim."""

import json
import tempfile
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from django.core.cache import cache
from django.test import override_settings
from django.urls import reverse

from gateway.config import get_openai_client, get_router, get_router_config
from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import INTEGRATION_TEST_BACKEND, GatewayIntegrationTestCase
from management.models import Token


@override_settings(
    AQUEDUCT_MODEL_RPM_LIMITS={"text-embedding-ada-002": 1}, AQUEDUCT_RATE_LIMIT_ENABLED=False
)
class PublicModelLimitTests(GatewayIntegrationTestCase):
    def setUp(self):
        super().setUp()
        if INTEGRATION_TEST_BACKEND != "openai":
            self.skipTest("These model names require the OpenAI test configuration")
        cache.clear()
        self.addCleanup(cache.clear)
        clock = patch(
            "django.utils.timezone.now",
            return_value=datetime(2026, 7, 2, 10, 0, 45, 500000, tzinfo=UTC),
        )
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        self.url = reverse("gateway:embeddings")

    def post_embedding(self, model, client=None, headers=None):
        return (client or self.client).post(
            self.url,
            data=json.dumps({"model": model, "input": "Hello"}),
            headers=headers or self.headers,
            content_type="application/json",
        )

    def test_aliases_and_different_tokens_share_one_public_model_budget(self):
        other_token, _ = self.create_new_user()
        self.assertEqual(self.post_embedding("embed").status_code, 200)
        with patch("litellm.Router.aembedding", new_callable=AsyncMock) as upstream:
            response = self.post_embedding(
                "text-embedding-ada-002", self.client_class(), _build_chat_headers(other_token)
            )
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response["Retry-After"], "15")
            upstream.assert_not_awaited()

    def test_excluded_tokens_cannot_consume_a_public_model_budget(self):
        other_token, _ = self.create_new_user()
        Token.objects.get(pk=1).user.profile.add_excluded_model("text-embedding-ada-002")
        self.assertEqual(self.post_embedding("embed").status_code, 404)
        self.assertEqual(
            self.post_embedding(
                "embed", self.client_class(), _build_chat_headers(other_token)
            ).status_code,
            200,
        )
        self.assertEqual(self.post_embedding("embed").status_code, 404)

    def test_unregistered_sdk_model_names_are_rejected_before_routing(self):
        with patch(
            "litellm.Router.aembedding",
            new_callable=AsyncMock,
            side_effect=AssertionError("Unregistered model reached the router"),
        ) as upstream:
            response = self.post_embedding("internal-deployment-id")
            self.assertEqual(response.status_code, 400)
            upstream.assert_not_awaited()

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={"text-embedding-ada-002": 2})
    def test_budget_resets_at_the_next_utc_minute(self):
        for _ in range(2):
            self.assertEqual(self.post_embedding("embed").status_code, 200)
        self.clock.return_value += timedelta(seconds=14)
        response = self.post_embedding("embed")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response["Retry-After"], "1")
        self.clock.return_value += timedelta(seconds=1)
        self.assertEqual(self.post_embedding("embed").status_code, 200)

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={})
    def test_unconfigured_models_do_not_touch_the_admission_cache(self):
        with patch("django.core.cache.cache.incr", side_effect=ConnectionError("unavailable")):
            self.assertEqual(self.post_embedding("embed").status_code, 200)

    def test_cache_failure_stops_the_request_before_upstream(self):
        with (
            patch("django.core.cache.cache.incr", side_effect=ConnectionError("unavailable")),
            patch("litellm.Router.aembedding", new_callable=AsyncMock) as upstream,
        ):
            response = self.post_embedding("embed")
            self.assertEqual(response.status_code, 500)
            upstream.assert_not_awaited()

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={"gpt-4.1-nano": 2, "dall-e-2": 2})
    def test_direct_openai_endpoints_share_public_budgets_with_legacy_deployment_ids(self):
        config = deepcopy(get_router_config())
        for model in config["model_list"]:
            model["model_info"]["id"] = f"private-{model['model_name']}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            path.write_text(json.dumps(config))
            with override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=path):
                for loader in (get_router, get_router_config, get_openai_client):
                    loader.cache_clear()
                    self.addCleanup(loader.cache_clear)
                for endpoint, payload in [
                    ("v1_responses", {"model": "main", "input": "Hello"}),
                    ("image_generation", {"model": "dall-e-2", "prompt": "A landscape"}),
                ]:
                    with self.subTest(endpoint=endpoint):
                        url = reverse(f"gateway:{endpoint}")
                        public = "gpt-4.1-nano" if endpoint == "v1_responses" else "dall-e-2"
                        for name, expected in [
                            (payload["model"], 200),
                            (f"private-{public}", 200),
                            (payload["model"], 429),
                            (f"private-{public}", 429),
                        ]:
                            response = self.client.post(
                                url,
                                data=json.dumps({**payload, "model": name}),
                                headers=self.headers,
                                content_type="application/json",
                            )
                            self.assertEqual(response.status_code, expected)

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={"gpt-4.1-nano": 1})
    def test_multi_deployment_responses_stay_pinned_and_share_the_public_budget(self):
        config = deepcopy(get_router_config())
        original = next(
            model for model in config["model_list"] if model["model_name"] == "gpt-4.1-nano"
        )
        original["model_info"]["id"] = "first-backend"
        second = deepcopy(original)
        second["model_info"] = {"id": "second-backend"}
        config["model_list"].append(second)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            path.write_text(json.dumps(config))
            with override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=path):
                for loader in (get_router, get_router_config, get_openai_client):
                    loader.cache_clear()
                    self.addCleanup(loader.cache_clear)
                with patch(
                    "litellm.Router.get_available_deployment",
                    side_effect=AssertionError(
                        "Direct Responses operations must not reselect a backend"
                    ),
                ) as select:
                    url = reverse("gateway:v1_responses")

                    def create(model):
                        return self.client.post(
                            url,
                            data=json.dumps({"model": model, "input": "Hello"}),
                            headers=self.headers,
                            content_type="application/json",
                        )

                    # Preserve the direct SDK's explicit-ID contract for ambiguous names.
                    self.assertEqual(create("gpt-4.1-nano").status_code, 404)
                    cache.clear()
                    response = create("first-backend")
                    self.assertEqual(response.status_code, 200)
                    response_url = reverse(
                        "gateway:v1_response", kwargs={"response_id": response.json()["id"]}
                    )
                    self.assertEqual(
                        self.client.get(response_url, headers=self.headers).status_code, 200
                    )
                    self.assertEqual(create("second-backend").status_code, 429)
                    select.assert_not_called()

    @override_settings(AQUEDUCT_MODEL_RPM_LIMITS={"gpt-4o-mini-tts": 1, "gpt-4.1-nano": 1})
    async def test_speech_and_streaming_chat_have_independent_model_budgets(self):
        for endpoint, payload in [
            ("speech", {"model": "voice", "input": "Hello", "voice": "alloy"}),
            (
                "chat_completions",
                {
                    "model": "main",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            ),
        ]:
            with self.subTest(endpoint=endpoint):
                url = reverse(f"gateway:{endpoint}")
                response = await self.async_client.post(
                    url,
                    data=json.dumps(payload),
                    headers=self.headers,
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 200)
                content = b"".join([chunk async for chunk in response.streaming_content])
                self.assertGreater(len(content), 0)
                response = await self.async_client.post(
                    url,
                    data=json.dumps(payload),
                    headers=self.headers,
                    content_type="application/json",
                )
                self.assertEqual(response.status_code, 429)
