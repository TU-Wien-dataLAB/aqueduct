import json
from http import HTTPStatus

from django.urls import reverse

from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase
from gateway.tests.utils.mock_server import MockConfig
from management.models import Request, Usage


class ImageGenerationEndpointTest(GatewayIntegrationTestCase):
    """Test the image generation endpoint."""

    model = "dall-e-2"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)
        cls.url = reverse("gateway:image_generation")

    def test_image_generation_endpoint_defaults(self):
        """Test basic image generation with valid parameters."""

        payload = {
            "model": self.model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.OK)

        response_json = response.json()
        self.assertIn("data", response_json, "Response should contain 'data' field")
        self.assertIsInstance(response_json["data"], list, "Data should be a list")
        self.assertGreater(len(response_json["data"]), 0, "Data should not be empty")

        # Check first image object structure
        img_data = response_json["data"][0]
        self.assertIn("b64_json", img_data, "Image data should contain 'b64_json' data")

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after image generation."
        )
        req = requests[0]
        self.assertIn(
            "images/generations", req.path, "Request endpoint should be for image generation."
        )
        self.assertIsInstance(req.token_usage, Usage)

    def test_image_generation_with_b64_json_response_format(self):
        """Test image generation with response format "b64_json"."""

        payload = {
            "model": self.model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
            "response_format": "b64_json",
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.OK)

        response_json = response.json()
        self.assertIn("data", response_json, "Response should contain 'data' field")
        self.assertIsInstance(response_json["data"], list, "Data should be a list")
        self.assertGreater(len(response_json["data"]), 0, "Data should not be empty")

        # Check first image object structure
        img_data = response_json["data"][0]
        self.assertIn("b64_json", img_data, "Image data should contain 'b64_json' field")
        self.assertIsInstance(img_data["b64_json"], str, "Image data should be a string")

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after image generation."
        )
        req = requests[0]
        self.assertIsInstance(req.token_usage, Usage)

    def test_image_generation_endpoint_missing_required_fields(self):
        """Test image generation endpoint with missing required fields."""

        payload = {
            "model": self.model,
            # Missing 'prompt' field
            "size": "256x256",
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)

    def test_image_generation_endpoint_empty_prompt(self):
        """Test image generation endpoint with empty prompt."""

        payload = {
            "model": self.model,
            "prompt": "",  # Empty prompt
            "n": 1,
            "size": "256x256",
        }

        with self.mock_server.patch_external_api(
            self.url,
            MockConfig(
                status_code=HTTPStatus.BAD_REQUEST,
                response_data={"error": "Invalid 'prompt': empty string"},
            ),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid 'prompt': empty string", response.json()["error"])

    def test_image_generation_endpoint_non_image_model(self):
        """Test image generation endpoint with a model that doesn't support image generation."""

        payload = {"model": "gpt-4.1-nano", "prompt": "A test image", "n": 1, "size": "256x256"}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Invalid value: 'gpt-4.1-nano'", response.json()["error"])

    def test_image_generation_endpoint_with_multiple_images(self):
        """Test image generation endpoint with multiple images (n parameter)."""

        payload = {
            "model": self.model,
            "prompt": "Generate multiple images of a sunset",
            "n": 2,
            "size": "256x256",
        }

        with self.mock_server.patch_external_api(
            self.url,
            MockConfig(
                status_code=HTTPStatus.OK,
                response_data={
                    "created": 1713833628,
                    "data": [
                        {
                            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                        },
                        {
                            "b64_json": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                        },
                    ],
                    "usage": {
                        "total_tokens": 100,
                        "input_tokens": 50,
                        "output_tokens": 50,
                        "input_tokens_details": {"text_tokens": 10, "image_tokens": 40},
                    },
                },
            ),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.OK)

        response_json = response.json()
        self.assertIn("data", response_json, "Response should contain 'data' field")
        # TODO: Should return 2, returns 1 img
        self.assertEqual(len(response_json["data"]), 2, "Should generate exactly 2 images")

        # Each image should have required fields
        for image in response_json["data"]:
            self.assertIn("b64_json", image, "Each image should contain 'b64_json' data")

    def test_image_generation_endpoint_invalid_size(self):
        """Test image generation endpoint with invalid size parameter."""

        payload = {"model": self.model, "prompt": "A test image", "size": "1x1"}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)

    def test_image_generation_endpoint_stream(self):
        """Test image generation endpoint with `stream=True`."""

        payload = {"model": self.model, "prompt": "A test image", "stream": True}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Aqueduct does not support image streaming.", response.json()["error"])

    def test_image_generation_with_extra_fields_in_body(self):
        """Extra fields in the request body are not allowed."""

        payload = {"model": self.model, "prompt": "A test image", "sth_extra": "Oh yeah"}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("Extra inputs are not permitted", response.json()["error"])
