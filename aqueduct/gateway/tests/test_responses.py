import json

from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth import get_user_model
from django.core.cache import caches

from gateway.tests.utils import _build_chat_headers, _read_streaming_response_lines
from gateway.tests.utils.base import GatewayIntegrationTestCase
from management.models import Request

User = get_user_model()


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

        self.assertIsNotNone(req.input_tokens)
        self.assertGreater(req.input_tokens, 0)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.output_tokens, 0)

        # Verify response was registered in cache
        response_id = response_json.get("id")
        self.assertIsNotNone(response_id, "Response ID should be present in response")

        # Check cache for response registration
        cache_key = f"response:{response_id}"
        cached_data = caches["default"].get(cache_key)
        if cached_data is not None:
            self.assertEqual(
                cached_data.get("model"), self.model, "Cached model name should match request model"
            )
            self.assertEqual(
                cached_data.get("email"),
                "me@example.com",
                "Cached user email should match current user",
            )
        else:
            self.fail(f"Response {response_id} should be registered in cache with key {cache_key}")

    @async_to_sync
    async def test_create_response_streaming(self):
        """
        Tests streaming response creation via POST /v1/responses with stream=True.
        Verifies that the endpoint returns a streaming response with proper format
        and logs the request to the database.
        """

        # Authenticate async client
        await sync_to_async(
            lambda: self.async_client.force_login(
                User.objects.get_or_create(username="Me", email="me@example.com")[0]
            )
        )()

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        # Streaming request payload
        payload = {
            "model": self.model,
            "input": [{"role": "user", "content": "Hello, how are you?"}],
            "max_output_tokens": 50,
            "stream": True,
        }

        response = await self.async_client.post(
            "/v1/responses",
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Collect all streamed lines
        streamed_lines = await _read_streaming_response_lines(response)
        self.assertGreater(
            len(streamed_lines), 0, "Should receive at least one streamed data chunk."
        )

        # Extract text and response ID from responses API streaming format
        # The responses API uses a different format than chat completions
        # Look for response.output_text.delta events with "text" fields
        # Also capture response ID from response.created event
        full_content = ""
        response_id = None
        for line in streamed_lines:
            try:
                data = json.loads(line)
                if data.get("type") == "response.created" and "response" in data:
                    response_id = data["response"].get("id")
                elif data.get("type") == "response.output_text.delta" and "delta" in data:
                    full_content += data["delta"]
            except (json.JSONDecodeError, KeyError):
                continue

        self.assertTrue(full_content, "Streamed content should not be empty.")

        # Verify content from streaming response
        self.assertIn("Hello", full_content, "Streaming response should contain greeting.")

        # Check that the database contains one request and endpoint matches
        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(
            len(requests),
            1,
            "There should be exactly one request after streaming response creation.",
        )
        req = requests[0]
        self.assertIn(
            "responses", req.path, "Request endpoint should be for responses (streaming)."
        )

        self.assertIsNotNone(req.input_tokens)
        self.assertGreater(req.input_tokens, 0)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.output_tokens, 0)

        # Verify response was registered in cache (for streaming)
        self.assertIsNotNone(
            response_id, "Response ID should be captured from streaming response.created event"
        )

        cache_key = f"response:{response_id}"
        cached_data = await sync_to_async(caches["default"].get)(cache_key)
        if cached_data is not None:
            self.assertEqual(
                cached_data.get("model"), self.model, "Cached model name should match request model"
            )
            self.assertEqual(
                cached_data.get("email"),
                "me@example.com",
                "Cached user email should match current user",
            )
        else:
            self.fail(f"Response {response_id} should be registered in cache with key {cache_key}")
