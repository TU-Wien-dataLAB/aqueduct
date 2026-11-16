import json

from gateway.tests.utils.base import ChatCompletionsBase
from management.models import Org


class ClientRouterSettingsTest(ChatCompletionsBase):
    """Body fields that LiteLLM reads as routing or credential settings must not take effect."""

    UNREACHABLE_API_BASE = "http://127.0.0.1:9"

    def _post(self, path: str, payload: dict):
        return self.client.post(
            path, data=json.dumps(payload), headers=self.headers, content_type="application/json"
        )

    def test_api_base_in_body_does_not_redirect_the_upstream_request(self):
        payloads = {
            "/chat/completions": {"model": self.model, "messages": self.MESSAGES},
            "/completions": {"model": self.model, "prompt": "Hi"},
            "/embeddings": {"model": "text-embedding-ada-002", "input": "Hi"},
            "/audio/speech": {"model": "gpt-4o-mini-tts", "input": "Hi", "voice": "alloy"},
        }
        for path, payload in payloads.items():
            with self.subTest(path=path):
                response = self._post(path, {**payload, "api_base": self.UNREACHABLE_API_BASE})
                self.assertEqual(response.status_code, 200)

    def test_extra_body_cannot_override_the_checked_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model("dall-e-2")
        org.save()

        payloads = {
            "/chat/completions": {"model": self.model, "messages": self.MESSAGES},
            "/completions": {"model": self.model, "prompt": "Hi"},
            "/embeddings": {"model": "text-embedding-ada-002", "input": "Hi"},
            "/audio/speech": {"model": "gpt-4o-mini-tts", "input": "Hi", "voice": "alloy"},
        }
        for path, payload in payloads.items():
            with self.subTest(path=path):
                response = self._post(path, {**payload, "extra_body": {"model": "dall-e-2"}})
                self.assertEqual(response.status_code, 200)
                # LiteLLM adapts text completions for chat models to chat/completions.
                upstream_path = "chat/completions" if path == "/completions" else path
                upstream_body = self.mock_server.get_last_request(upstream_path)
                self.assertEqual(upstream_body["model"], payload["model"])

    def test_mock_response_in_body_is_not_returned(self):
        response = self._post(
            self.url,
            {"model": self.model, "messages": self.MESSAGES, "mock_response": "not from the model"},
        )

        self.assertEqual(response.status_code, 200, response.content)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertNotEqual(content, "not from the model")

    def test_request_fallbacks_cannot_reach_an_excluded_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model("dall-e-2")
        org.save()

        response = self._post(
            self.url,
            {
                "model": self.model,
                "messages": self.MESSAGES,
                "fallbacks": ["dall-e-2"],
                "mock_testing_fallbacks": True,
            },
        )

        self.assertEqual(response.status_code, 200, response.content)
        upstream_body = self.mock_server.get_last_request("chat/completions")
        self.assertEqual(upstream_body["model"], self.model)

    def test_provider_specific_fields_still_reach_the_upstream(self):
        response = self._post(
            self.url,
            {
                "model": self.model,
                "messages": self.MESSAGES,
                "top_k": 5,
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_body": {"top_k": 7, "min_p": 0.1},
            },
        )

        self.assertEqual(response.status_code, 200, response.content)
        upstream_body = self.mock_server.get_last_request("chat/completions")
        self.assertEqual(upstream_body["top_k"], 5)
        self.assertEqual(upstream_body["min_p"], 0.1)
        self.assertEqual(upstream_body["chat_template_kwargs"], {"enable_thinking": False})
