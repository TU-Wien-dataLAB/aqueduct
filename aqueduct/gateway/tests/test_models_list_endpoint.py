from unittest.mock import patch

from gateway.config import get_router_config
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Org, Request


class ListModelsIntegrationTest(GatewayIntegrationTestCase):
    def _send_model_list_request(self):
        return self.client.get("/models", content_type="application/json", headers=self.headers)

    def test_list_models(self):
        """
        Sends a request to list available models from the vLLM server using the Django test client.
        After the request, checks that the database contains one request and the endpoint matches.
        """
        response = self._send_model_list_request()

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        self.assertIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")

    def test_model_group_info_exposes_context_window(self):
        """The LiteLLM /model_group/info endpoint exposes token limits, so
        LiteLLM-aware clients read the context window from the gateway."""

        config = {
            "model_list": [
                {
                    "model_name": self.model,
                    "litellm_params": {"model": f"openai/{self.model}"},
                    "model_info": {
                        "id": self.model,
                        "max_tokens": 262144,
                        "max_output_tokens": 32768,
                        "supports_vision": True,
                    },
                },
                {
                    "model_name": "explicit-model",
                    "litellm_params": {"model": "openai/explicit-model"},
                    "model_info": {"max_input_tokens": 999999, "max_tokens": 262144},
                },
                {
                    "model_name": "no-info-model",
                    "litellm_params": {"model": "openai/no-info-model"},
                },
            ]
        }
        with patch("gateway.views.models.get_router_config", return_value=config):
            response = self.client.get(
                "/model_group/info", content_type="application/json", headers=self.headers
            )

        self.assertEqual(response.status_code, 200)
        entries = {entry["model_group"]: entry for entry in response.json()}
        info = entries[self.model]["model_info"]
        # max_input_tokens is derived from max_tokens (the context length)
        self.assertEqual(info["max_input_tokens"], 262144)
        self.assertEqual(info["max_output_tokens"], 32768)
        self.assertTrue(info["supports_vision"])
        # An explicit max_input_tokens in the config is never overwritten
        self.assertEqual(entries["explicit-model"]["model_info"]["max_input_tokens"], 999999)
        # Models without token limits get no token fields at all
        self.assertNotIn("max_input_tokens", entries["no-info-model"]["model_info"])

    def test_list_models_with_invalid_token(self):
        """
        Sends a request to list available models from the vLLM server with an invalid API key.
        Expects an authentication error (401 or 403).
        """
        # Prepare headers with an invalid token
        headers = {
            "HTTP_AUTHORIZATION": "Bearer invalid-token-123",
            "CONTENT_TYPE": "application/json",
        }

        # No payload needed for model listing
        response = self.client.get("/models", content_type="application/json", headers=headers)

        # Should be 401 Unauthorized or 403 Forbidden
        self.assertIn(
            response.status_code,
            [401, 403],
            f"Expected 401 or 403 for invalid token, "
            f"got {response.status_code}: {response.content}",
        )

        # There should be no request recorded in the database
        # (or possibly one, depending on implementation)
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 0, "There should be no request recorded for invalid token.")

    def test_list_excluded_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()
        assert len(org.excluded_models) == 1

        router_config = get_router_config()
        model_list: list[dict] = router_config["model_list"]

        response = self._send_model_list_request()

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        self.assertEqual(len(model_ids), len(model_list) - 1)
        self.assertNotIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")
