import json

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request


class ResponsesIntegrationTest(GatewayIntegrationTestCase):
    def test_create_response_basic(self):
        """
        Tests basic response creation via POST /v1/responses.
        Verifies that the endpoint exists and returns a 200 status code.
        """
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        # Basic request payload for responses API
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": "Hello, how are you?"}],
            "max_output_tokens": 50,
        }

        response = self.client.post(
            "/v1/responses",
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json",
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        self.assertIn("id", response_json)
        self.assertIn("object", response_json)
        self.assertEqual(response_json["object"], "response")
        self.assertIn("output", response_json)
        self.assertIsInstance(response_json["output"], list)

        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after response creation."
        )
        req = requests[0]
        self.assertIn("responses", req.path, "Request endpoint should be for responses.")
