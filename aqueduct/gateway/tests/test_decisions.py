import json

from django.urls import reverse

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request

_SAMPLE_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7w"
    "AAAABJRU5ErkJggg=="
)


class DecisionsIntegrationTest(GatewayIntegrationTestCase):
    """Integration tests for POST /v1/decisions (non-streaming)."""

    model = "gpt-6-luna"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.url = reverse("gateway:v1_decisions")

    async def test_create_decision_predicate(self):
        """A predicate question returns a probability in the answers array."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "input": "Inspect the product in this photo.",
            "questions": [
                {
                    "type": "predicate",
                    "name": "visible_damage",
                    "instructions": "Does the product have visible damage?",
                }
            ],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        self.assertIn("answers", response_json)
        self.assertIsInstance(response_json["answers"], list)
        self.assertEqual(response_json["answers"][0]["name"], "visible_damage")
        self.assertEqual(response_json["answers"][0]["type"], "predicate")
        self.assertIn("probability", response_json["answers"][0])

        requests = [r async for r in Request.objects.all()]
        self.assertEqual(len(requests), 1, "Exactly one request should be logged.")
        req = requests[0]
        self.assertIn("decisions", req.path, "Request endpoint should be for decisions.")
        self.assertEqual(req.model, self.model)
        self.assertGreater(req.input_tokens, 0)
        self.assertGreater(req.output_tokens, 0)

    async def test_create_decision_forwards_upstream_body(self):
        """The request body (model, input, questions) is forwarded to the upstream API."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "input": "Which department handles refunds?",
            "questions": [
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which department should handle this complaint?",
                    "choices": [
                        {"value": "billing", "description": "Payments, invoices, and refunds."},
                        {"value": "technical", "description": "Problems using the product."},
                    ],
                }
            ],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)

        forwarded = self.mock_server.get_last_request("decisions")
        self.assertIsNotNone(forwarded, "The decisions request should reach the upstream mock.")
        self.assertEqual(forwarded["model"], self.model)
        self.assertEqual(forwarded["input"], payload["input"])
        self.assertEqual(forwarded["questions"], payload["questions"])

    async def test_decision_with_images(self):
        """Text + image input (input_text / input_image parts) is forwarded upstream."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Inspect the product in this photo."},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{_SAMPLE_IMAGE_B64}",
                        },
                    ],
                }
            ],
            "questions": [
                {
                    "type": "predicate",
                    "name": "visible_damage",
                    "instructions": "Does the product have visible damage?",
                }
            ],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("answers", response.json())

    async def test_decision_alias_resolved(self):
        """Requesting the 'decision' alias resolves to gpt-6-luna upstream."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": "decision",
            "input": "Which department handles refunds?",
            "questions": [
                {
                    "type": "choice",
                    "name": "department",
                    "instructions": "Which department should handle this complaint?",
                    "choices": [
                        {"value": "billing", "description": "Payments, invoices, and refunds."},
                        {"value": "technical", "description": "Problems using the product."},
                    ],
                }
            ],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("answers", response.json())

        req = await Request.objects.aget()
        self.assertEqual(req.model, "gpt-6-luna")

    async def test_decision_missing_model(self):
        """A request without a model returns a 400 error."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "input": "Inspect the product.",
            "questions": [{"type": "predicate", "name": "d", "instructions": "Damage?"}],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)

    async def test_decision_unknown_model(self):
        """An unknown model returns a 404 (via oai_client_from_body)."""
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": "nonexistent-model",
            "input": "Inspect the product.",
            "questions": [{"type": "predicate", "name": "d", "instructions": "Damage?"}],
        }

        response = await self.async_client.post(
            self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
        )

        self.assertEqual(response.status_code, 404)
