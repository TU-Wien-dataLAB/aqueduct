import json
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from gateway.config import get_openai_client, get_router, get_router_config
from gateway.tests.utils import _build_chat_headers
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request, Usage


class ImageGenerationEndpointTest(GatewayIntegrationTestCase):
    """Test the image generation endpoint."""

    model = "dall-e-2"
    url = "/images/generations"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.headers = _build_chat_headers(cls.AQUEDUCT_ACCESS_TOKEN)

    def test_image_generation_endpoint_defaults(self):
        """Test basic image generation with valid parameters."""

        payload = {
            "model": self.model,
            "prompt": "A beautiful landscape with mountains and a lake",
            "size": "256x256",
        }

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

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

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

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

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_image_generation_endpoint_empty_prompt(self):
        """Test image generation endpoint with empty prompt."""

        payload = {
            "model": self.model,
            "prompt": "",  # Empty prompt
            "n": 1,
            "size": "256x256",
        }

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_image_generation_endpoint_non_image_model(self):
        """Test image generation endpoint with a model that doesn't support image generation."""

        payload = {"model": "gpt-4.1-nano", "prompt": "A test image", "n": 1, "size": "256x256"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Incompatible model 'gpt-4.1-nano'!", response.json()["error"])

    def test_image_generation_endpoint_with_multiple_images(self):
        """Test image generation endpoint with multiple images (n parameter)."""

        payload = {
            "model": self.model,
            "prompt": "Generate multiple images of a sunset",
            "n": 2,
            "size": "256x256",
        }

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)

        response_json = response.json()
        self.assertIn("data", response_json, "Response should contain 'data' field")
        self.assertEqual(len(response_json["data"]), 2, "Should generate exactly 2 images")

        # Each image should have required fields
        for image in response_json["data"]:
            self.assertIn("b64_json", image, "Each image should contain 'b64_json' data")

    def test_image_generation_endpoint_invalid_size(self):
        """Test image generation endpoint with invalid size parameter."""

        payload = {"model": self.model, "prompt": "A test image", "size": "1x1"}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)

    def test_image_generation_endpoint_stream(self):
        """Test image generation endpoint with `stream=True`."""

        payload = {"model": self.model, "prompt": "A test image", "stream": True}

        response = self.client.post(
            self.url,
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Aqueduct does not support image streaming.", response.json()["error"])

    def test_image_generation_with_alias(self):
        """Test that image generation endpoint correctly resolves model aliases."""

        # Mock config with alias for image generation model
        mock_config = {
            "model_list": [
                {
                    "model_name": self.model,
                    "litellm_params": {
                        "model": f"openai/{self.model}",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"mode": "image_generation", "aliases": ["image", "dalle"]},
                }
            ]
        }

        # Clear both caches to ensure fresh state
        get_router_config.cache_clear()
        get_router.cache_clear()
        get_openai_client.cache_clear()

        mock_image_object = Image(
            b64_json="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            revised_prompt="A beautiful landscape with mountains and a lake",
        )
        mock_image_response = ImagesResponse(data=[mock_image_object], created=123456789)

        with patch("gateway.config.get_router_config", return_value=mock_config):
            with patch("gateway.views.image_generation.get_router") as mock_router:
                with patch(
                    "gateway.views.image_generation.litellm.get_llm_provider"
                ) as mock_get_llm_provider:
                    deployment_mock = MagicMock()
                    deployment_mock.litellm_params.model = f"openai/{self.model}"
                    mock_router.get_deployment = MagicMock(return_value=deployment_mock)

                    # Mock litellm.get_llm_provider to return the relay model and provider
                    mock_get_llm_provider.return_value = (self.model, "openai", None, None)

                    with patch("gateway.views.image_generation.get_openai_client") as mock_client:
                        # Mock the client's images.generate method to return our mock response
                        mock_openai_client = AsyncMock()
                        mock_openai_client.images.generate.return_value = mock_image_response
                        mock_client.return_value = mock_openai_client

                        payload = {
                            "model": "image",  # Using alias instead of actual model name
                            "prompt": "A beautiful landscape with mountains and a lake",
                            "size": "256x256",
                        }

                        response = self.client.post(
                            "/images/generations",
                            data=json.dumps(payload),
                            headers=_build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN),
                            content_type="application/json",
                        )

                        # Should return 200 with alias resolution
                        self.assertEqual(response.status_code, 200)

                        response_json = response.json()
                        self.assertIn("data", response_json, "Response should contain 'data' field")
                        self.assertIsInstance(response_json["data"], list, "Data should be a list")
                        self.assertGreater(
                            len(response_json["data"]), 0, "Data should not be empty"
                        )

                        # Check first image object structure
                        img_data = response_json["data"][0]
                        self.assertIn(
                            "b64_json", img_data, "Image data should contain 'b64_json' field"
                        )
                        self.assertIsInstance(
                            img_data["b64_json"], str, "Image data should be a string"
                        )

                        # Check that the database contains one request
                        requests = list(Request.objects.all())
                        self.assertEqual(
                            len(requests),
                            1,
                            "There should be exactly one request after image generation.",
                        )
