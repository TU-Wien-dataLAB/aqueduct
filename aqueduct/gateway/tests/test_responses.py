import json
from unittest.mock import AsyncMock, MagicMock, patch

from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from django.core.cache import caches

from gateway.tests.utils import _build_chat_headers, _read_streaming_response_lines
from gateway.tests.utils.base import GatewayIntegrationTestCase
from gateway.views.utils import register_response_in_cache
from management.models import Request, Token

User = get_user_model()


class ResponsesIntegrationTest(GatewayIntegrationTestCase):
    async def test_create_response_basic(self):
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

        response = await self.async_client.post(
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

        requests = await sync_to_async(list)(Request.objects.all())
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
        self.assertEqual(
            cached_data.get("model"), self.model, "Cached model name should match request model"
        )
        self.assertEqual(
            cached_data.get("email"),
            "me@example.com",
            "Cached user email should match current user",
        )

        # Test GET response endpoint
        get_response = await self.async_client.get(f"/v1/responses/{response_id}", headers=headers)
        self.assertEqual(
            get_response.status_code,
            200,
            f"GET response should return 200, got {get_response.status_code}: {get_response.content}",
        )
        get_data = get_response.json()
        self.assertIn("id", get_data)
        self.assertEqual(get_data["id"], response_id)
        self.assertEqual(get_data["object"], "response")

        # Test GET response input_items endpoint
        input_items_response = await self.async_client.get(
            f"/v1/responses/{response_id}/input_items", headers=headers
        )
        self.assertEqual(
            input_items_response.status_code,
            200,
            f"GET input_items should return 200, got {input_items_response.status_code}: {input_items_response.content}",
        )
        input_items_data = input_items_response.json()
        self.assertIsInstance(input_items_data, dict)
        self.assertIn("data", input_items_data)
        self.assertIsInstance(input_items_data["data"], list)

        # Test DELETE response endpoint
        delete_response = await self.async_client.delete(
            f"/v1/responses/{response_id}", headers=headers
        )
        self.assertEqual(
            delete_response.status_code,
            200,
            f"DELETE response should return 200, got {delete_response.status_code}: {delete_response.content}",
        )

        # Verify response is deleted - GET should now return 404
        verify_get_response = await self.async_client.get(
            f"/v1/responses/{response_id}", headers=headers
        )
        self.assertEqual(
            verify_get_response.status_code,
            404,
            f"GET after DELETE should return 404, got {verify_get_response.status_code}: {verify_get_response.content}",
        )

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

        # Test GET response endpoint for streaming response
        get_response = await self.async_client.get(f"/v1/responses/{response_id}", headers=headers)
        self.assertEqual(
            get_response.status_code,
            200,
            f"GET response should return 200, got {get_response.status_code}: {get_response.content}",
        )
        get_data = get_response.json()
        self.assertIn("id", get_data)
        self.assertEqual(get_data["id"], response_id)
        self.assertEqual(get_data["object"], "response")

        # Test GET response input_items endpoint for streaming response
        input_items_response = await self.async_client.get(
            f"/v1/responses/{response_id}/input_items", headers=headers
        )
        self.assertEqual(
            input_items_response.status_code,
            200,
            f"GET input_items should return 200, got {input_items_response.status_code}: {input_items_response.content}",
        )
        input_items_data = input_items_response.json()
        self.assertIsInstance(input_items_data, dict)
        self.assertIn("data", input_items_data)
        self.assertIsInstance(input_items_data["data"], list)

        # Test DELETE response endpoint for streaming response
        delete_response = await self.async_client.delete(
            f"/v1/responses/{response_id}", headers=headers
        )
        self.assertEqual(
            delete_response.status_code,
            200,
            f"DELETE response should return 200, got {delete_response.status_code}: {delete_response.content}",
        )

        # Verify streaming response is deleted - GET should now return 404
        verify_get_response = await self.async_client.get(
            f"/v1/responses/{response_id}", headers=headers
        )
        self.assertEqual(
            verify_get_response.status_code,
            404,
            f"GET after DELETE should return 404, got {verify_get_response.status_code}: {verify_get_response.content}",
        )

    def test_get_response_invalid_id(self):
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        response = self.client.get("/v1/responses/invalid-id", headers=headers)
        self.assertEqual(
            response.status_code, 404, f"Expected 404 Not Found, got {response.status_code}"
        )

    def test_delete_response_invalid_id(self):
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        response = self.client.delete("/v1/responses/invalid-id", headers=headers)
        self.assertEqual(
            response.status_code, 404, f"Expected 404 Not Found, got {response.status_code}"
        )

    def test_input_items_response_invalid_id(self):
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        response = self.client.get("/v1/responses/invalid-id/input_items", headers=headers)
        self.assertEqual(
            response.status_code, 404, f"Expected 404 Not Found, got {response.status_code}"
        )

    def test_user_response_isolation_via_id(self):
        """
        Test that two distinct users cannot access each other's responses via ID.
        Tests both GET, DELETE and input_items endpoints.
        Follows the flow: user1 creates response -> user1 accesses -> user2 tries and fails
        """
        # Mock OpenAI client for response creation
        with patch("gateway.views.utils.get_openai_client") as mock_client:
            mock_openai_client = AsyncMock()

            response_id = "resp_test123"
            register_response_in_cache(response_id, model=self.model, email="me@example.com")

            # Mock response for retrieve/delete/input_items
            mock_retrieve_response = MagicMock()
            mock_retrieve_response.model_dump.return_value = {
                "id": "resp_test123",
                "status": "completed",
            }
            mock_openai_client.responses.retrieve.return_value = mock_retrieve_response
            mock_openai_client.responses.delete.return_value = mock_retrieve_response
            mock_openai_client.responses.input_items.list.return_value = mock_retrieve_response

            mock_client.return_value = mock_openai_client

            # Step 1: User1 (me@example.com) can access their own response
            headers_me = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

            # Step 2: User1 can access their own response
            get_response_me = self.client.get(f"/v1/responses/{response_id}", headers=headers_me)
            self.assertEqual(
                get_response_me.status_code,
                200,
                f"User me@example.com should access their own response, got {get_response_me.status_code}",
            )

            # Step 3: User2 (someone@example.com) cannot access User1's response
            # Create a token for someone@example.com
            user_someone = User.objects.get(email="someone@example.com")
            token_someone = Token(name="TestToken", user=user_someone)
            token_someone_key = token_someone._set_new_key()
            token_someone.save()

            headers_someone = _build_chat_headers(token_someone_key)

            # Test GET endpoint isolation
            get_response_someone = self.client.get(
                f"/v1/responses/{response_id}", headers=headers_someone
            )
            self.assertEqual(
                get_response_someone.status_code,
                404,
                f"User someone@example.com should not access me@example.com's response, got {get_response_someone.status_code}",
            )

            # Test DELETE endpoint isolation
            delete_response_someone = self.client.delete(
                f"/v1/responses/{response_id}", headers=headers_someone
            )
            self.assertEqual(
                delete_response_someone.status_code,
                404,
                f"User someone@example.com should not delete me@example.com's response, got {delete_response_someone.status_code}",
            )

            # Test input_items endpoint isolation
            input_items_response = self.client.get(
                f"/v1/responses/{response_id}/input_items", headers=headers_someone
            )
            self.assertEqual(
                input_items_response.status_code,
                404,
                f"User someone@example.com should not access me@example.com's input_items, got {input_items_response.status_code}",
            )

            # Clean up - User1 should be able to delete their own response
            delete_response_me = self.client.delete(
                f"/v1/responses/{response_id}", headers=headers_me
            )
            self.assertEqual(
                delete_response_me.status_code,
                200,
                "User1 should be able to delete their own response",
            )
