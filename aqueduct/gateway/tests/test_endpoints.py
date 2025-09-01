import json
from unittest.mock import patch

from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth import get_user_model
from django.test import override_settings
from openai.types.chat import ChatCompletion

from gateway.router import get_router_config
from gateway.tests.utils import _build_chat_headers, _build_chat_payload, \
    _read_streaming_response_lines, _parse_streamed_content_pieces
from gateway.tests.utils.base import GatewayIntegrationTestCase, INTEGRATION_TEST_BACKEND, ROUTER_CONFIG
from management.models import Request, UserProfile, ServiceAccount, Team, Org

User = get_user_model()


class EmbeddingTest(GatewayIntegrationTestCase):
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "text-embedding-ada-002"

    @override_settings(RELAY_REQUEST_TIMEOUT=5)
    def test_embeddings(self):
        """
        Sends a simple embeddings request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... Requires GatewayIntegrationTestCase to manage multiple servers!")

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        assert self.model in ROUTER_CONFIG
        payload = {
            "model": self.model,
            "input": ["The quick brown fox jumps over the lazy dog."]
        }
        endpoint = f"/embeddings"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        response_json = response.json()
        print(f"\nEmbeddings response: {response_json}")

        # OpenAI-style embeddings response should have 'data' and 'embedding' fields
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)
        embedding_obj = response_json["data"][0]
        self.assertIn("embedding", embedding_obj)
        self.assertIsInstance(embedding_obj["embedding"], list)
        self.assertGreater(len(embedding_obj["embedding"]), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after embeddings.")
        req = requests[0]
        self.assertIn("embeddings", req.path, "Request endpoint should be for embeddings.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertEqual(req.output_tokens, 0, "output_tokens should be 0")


class ChatCompletionsIntegrationTest(GatewayIntegrationTestCase):
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a short poem!"}
    ]

    def _build_chat_completion_request(self, messages, stream=False, **payload_kwargs) -> dict:
        """
        Helper to build headers, payload, and endpoint for chat completion requests.
        """
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        payload = _build_chat_payload(self.model, messages, stream=stream, **payload_kwargs)
        endpoint = f"/chat/completions"
        return dict(path=endpoint, data=json.dumps(payload), headers=headers)

    def _send_chat_completion(self, messages, **payload_kwargs):
        """
        Helper to send a chat completion request (non-streaming) using Django test client.
        """
        request = self._build_chat_completion_request(messages, stream=False, **payload_kwargs)
        return self.client.post(**request, content_type="application/json")

    async def _send_chat_completion_streaming(self, messages, **payload_kwargs):
        """
        Helper to send a streaming chat completion request using Django async test client.
        """
        request = self._build_chat_completion_request(messages, stream=True, **payload_kwargs)
        response = await self.async_client.post(**request, content_type="application/json")
        return response

    def test_chat_completion(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        # Parse the response as JSON and convert to OpenAI ChatCompletion type for compatibility
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)

        print(f"\nChat completion response: {chat_completion}")

        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertIsInstance(chat_completion.choices, list)
        self.assertGreater(len(chat_completion.choices), 0)

        first_choice = chat_completion.choices[0]
        self.assertTrue(hasattr(first_choice, 'message'))
        self.assertIsNotNone(first_choice.message)
        self.assertTrue(hasattr(first_choice.message, 'content'))
        self.assertIsNotNone(first_choice.message.content)

        response_text = first_choice.message.content.strip()
        print(response_text)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

    @override_settings(TIKA_SERVER_URL=None)
    def test_chat_completion_base64_file_input(self):
        """
        Sends a chat completion request with base64 encoded file input.
        Tests the file upload functionality using base64 encoding.
        """
        import base64

        # Create a simple text file content and encode it as base64
        file_content = b"This is a test file content for base64 encoding."
        file_base64 = base64.b64encode(file_content).decode('utf-8')

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in this file?"},
                     {"type": "file",
                      "file": {
                          "filename": "test.txt",
                          "file_data": f"data:text/plain;base64,{file_base64}"
                      }
                      }]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        with patch('gateway.views.decorators.extract_text_with_tika',
                   return_value="This is a test file content for base64 encoding."):
            response = self.client.post(
                endpoint,
                data=json.dumps(payload),
                headers=headers,
                content_type="application/json"
            )

        self.assertEqual(response.status_code, 200,
                         f"Expected 200 OK, got {response.status_code}: {response.content}")
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content
        if content is not None:
            content = content.strip()
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after base64 file input.")
        req = requests[0]
        self.assertIn("chat/completions", req.path,
                      "Request endpoint should be chat completion for base64 file input.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for base64 file input")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 for base64 file input")

    def test_chat_completion_file_id_input(self):
        """
        Sends a chat completion request with file_id input.
        Tests the file upload functionality using a file ID from the files API.
        """
        from django.core.files.uploadedfile import SimpleUploadedFile

        # First, upload a file using the files API
        file_content = b'{"custom_id": "test_file_id_input"}\n'
        upload_file = SimpleUploadedFile("test_file_id.jsonl", file_content, content_type="application/jsonl")

        # Prepare headers for file upload (remove Content-Type for multipart)
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers.pop("Content-Type", None)

        # Upload the file
        upload_response = self.client.post(
            "/files",
            {"file": upload_file, "purpose": "user_data"},
            headers=headers,
        )
        self.assertEqual(upload_response.status_code, 200, f"File upload failed: {upload_response.json()}")
        upload_data = upload_response.json()
        file_id = upload_data["id"]

        # Now use the file_id in a chat completion request
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in this file?"},
                     {"type": "file",
                      "file": {
                          "file_id": file_id
                      }
                      }]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        with patch('gateway.views.decorators.extract_text_with_tika',
                   return_value="This is a test file content for base64 encoding."):
            response = self.client.post(
                endpoint,
                data=json.dumps(payload),
                headers=headers,
                content_type="application/json"
            )

        self.assertEqual(response.status_code, 200,
                         f"Expected 200 OK, got {response.status_code}: {response.content}")
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 2,
                         "There should be exactly two requests after file_id input (file upload + chat completion).")

        # Get the chat completion request (should be the one with /chat/completions path)
        chat_request = next(r for r in requests if "chat/completions" in r.path)
        req = chat_request
        self.assertIn("chat/completions", req.path,
                      "Request endpoint should be chat completion for file_id input.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for file_id input")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 for file_id input")

    def test_chat_completion_file_id_not_found(self):
        """
        Sends a chat completion request with a non-existent file_id.
        Should raise a 404 error.
        """
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in this file?"},
                     {"type": "file",
                      "file": {
                          "file_id": "non-existent-file-id"
                      }
                      }]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 404,
                         f"Expected 404 Not Found, got {response.status_code}: {response.content}")

    def test_chat_completion_file_id_different_user(self):
        """
        Sends a chat completion request with a file_id that was created by a different user.
        Should raise a 404 error.
        """
        from django.core.files.uploadedfile import SimpleUploadedFile
        from management.models import Token

        # First, upload a file using the files API with the default user
        file_content = b'{"custom_id": "test_file_id_input"}\n'
        upload_file = SimpleUploadedFile("test_file_id.jsonl", file_content, content_type="application/jsonl")

        # Prepare headers for file upload (remove Content-Type for multipart)
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers.pop("Content-Type", None)

        # Upload the file
        upload_response = self.client.post(
            "/files",
            {"file": upload_file, "purpose": "user_data"},
            headers=headers,
        )
        self.assertEqual(upload_response.status_code, 200, f"File upload failed: {upload_response.json()}")
        upload_data = upload_response.json()
        file_id = upload_data["id"]

        UPDATED_ACCESS_TOKEN = self.create_new_user()
        headers = _build_chat_headers(UPDATED_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in this file?"},
                     {"type": "file",
                      "file": {
                          "file_id": file_id
                      }
                      }]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        with patch('gateway.views.decorators.extract_text_with_tika',
                   return_value="This is a test file content for base64 encoding."):
            response = self.client.post(
                endpoint,
                data=json.dumps(payload),
                headers=headers,
                content_type="application/json"
            )

        self.assertEqual(response.status_code, 404,
                         f"Expected 404 Not Found, got {response.status_code}: {response.content}")

    def test_chat_completion_base64_file_size_limit_individual(self):
        """
        Tests that individual files uploaded via base64 are rejected if they exceed 10MB.
        """
        import base64

        # Create a file content that exceeds 10MB (10MB + 1 byte)
        file_size = 10 * 1024 * 1024 + 1  # 10MB + 1 byte
        file_content = b"x" * file_size
        file_base64 = base64.b64encode(file_content).decode('utf-8')

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in this file?"},
                     {"type": "file",
                      "file": {
                          "filename": "large_file.txt",
                          "file_data": f"data:text/plain;base64,{file_base64}"
                      }
                      }]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400,
                         f"Expected 400 Bad Request for file size > 10MB, got {response.status_code}: {response.content}")

    def test_chat_completion_base64_file_size_limit_total(self):
        """
        Tests that total files uploaded via base64 are rejected if they exceed 32MB in total.
        """
        import base64

        # Create multiple files that together exceed 32MB
        # Each file is 11MB, so 3 files = 33MB total
        individual_file_size = 11 * 1024 * 1024  # 11MB each
        file_content = b"x" * individual_file_size
        file_base64 = base64.b64encode(file_content).decode('utf-8')

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What's in these files?"},
                     {"type": "file",
                      "file": {
                          "filename": "large_file_1.txt",
                          "file_data": f"data:text/plain;base64,{file_base64}"
                      }},
                     {"type": "file",
                      "file": {
                          "filename": "large_file_2.txt",
                          "file_data": f"data:text/plain;base64,{file_base64}"
                      }},
                     {"type": "file",
                      "file": {
                          "filename": "large_file_3.txt",
                          "file_data": f"data:text/plain;base64,{file_base64}"
                      }}
                 ]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }

        endpoint = "/chat/completions"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400,
                         f"Expected 400 Bad Request for total file size > 32MB, got {response.status_code}: {response.content}")

    @override_settings(RELAY_REQUEST_TIMEOUT=0.1)
    def test_chat_completion_timeout(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(
            response.status_code, 504,
            f"Expected 504 Gateway Timeout, got {response.status_code}: {response.content}"
        )

    @override_settings(STREAM_REQUEST_TIMEOUT=5)
    @async_to_sync
    async def test_chat_completion_streaming(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # For some reason authentication does not work in async test case...
        await sync_to_async(lambda: self.async_client.force_login(
            User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        response = await self._send_chat_completion_streaming(self.MESSAGES)

        # Should be a StreamingHttpResponse with status 200
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Collect all streamed lines (each line is a data: ... event)
        streamed_lines = await _read_streaming_response_lines(response)

        self.assertGreater(len(streamed_lines), 0, "Should receive at least one streamed data chunk.")

        # Parse each chunk as JSON and collect content pieces
        content_pieces = _parse_streamed_content_pieces(streamed_lines)
        full_content = "".join(content_pieces).strip()
        print(f"Full streamed content: {full_content}")
        self.assertTrue(full_content, "Streamed content should not be empty.")

        # Check that the database contains one request and endpoint matches
        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(len(requests), 1, "There should be exactly one request after streaming chat completion.")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion (streaming).")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 (streaming)")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 (streaming)")

    @override_settings(RELAY_REQUEST_TIMEOUT=0.001)
    @async_to_sync
    async def test_chat_completion_streaming_relay_request_timeout(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # For some reason authentication does not work in async test case...
        await sync_to_async(lambda: self.async_client.force_login(
            User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        response = await self._send_chat_completion_streaming(self.MESSAGES)

        self.assertEqual(
            response.status_code, 504,
            f"Expected 504 Gateway Timeout, got {response.status_code}"
        )

    def test_chat_completion_excluded_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()
        assert len(org.excluded_models) == 1

        response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(response.status_code, 404,
                         f"Expected 404 Model not found, got {response.status_code}: {response.content}")

    def test_chat_completion_unknown_model(self):
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        payload = {
            "model": "unknown-model",
            "messages": self.MESSAGES,
        }
        response = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400,
                         f"Expected 400 Bad Request, got {response.status_code}: {response.content}")

    @override_settings(RELAY_REQUEST_TIMEOUT=5)
    def test_chat_completion_schema_generation(self):
        """
        Sends a chat completion request with a JSON schema, non-streaming.
        Verifies the response content adheres to the schema and logs the request.
        """
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        json_schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["greeting", "count"]
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce JSON output based on a schema."},
                {"role": "user", "content": "Generate JSON matching the provided schema."}
            ],
            # { "type": "json_schema", "json_schema": {...} }
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": json_schema}
            },
            "max_tokens": 50,
            "temperature": 0.0
        }
        endpoint = "/chat/completions"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200,
                         f"Expected 200 OK, got {response.status_code}: {response.content}")
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        result = json.loads(content)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(json_schema["properties"].keys()))
        self.assertIsInstance(result["greeting"], str)
        self.assertIsInstance(result["count"], int)

        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after schema generation.")
        req = requests[0]
        self.assertIn("chat/completions", req.path,
                      "Request endpoint should be for chat completion schema generation.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for schema generation")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 for schema generation")

    def test_chat_completion_multimodal_input(self):
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... Requires GatewayIntegrationTestCase to manage multiple servers!")

        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "user",
                 "content": [
                     {"type": "text", "text": "What’s in this image?"},
                     {"type": "image_url", "image_url": {"url": image_url}},
                 ]}
            ],
            "max_tokens": 50,
            "temperature": 0.0
        }
        endpoint = "/chat/completions"
        response = self.client.post(
            endpoint,
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200,
                         f"Expected 200 OK, got {response.status_code}: {response.content}")
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after multimodal input.")
        req = requests[0]
        self.assertIn("chat/completions", req.path,
                      "Request endpoint should be chat completion for multimodal input.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for multimodal input")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 for multimodal input")

    @override_settings(STREAM_REQUEST_TIMEOUT=5)
    @async_to_sync
    async def test_chat_completion_schema_generation_streaming(self):
        """
        Sends a streaming chat completion request with a JSON schema.
        Verifies the streamed content adheres to the schema and logs the request.
        """
        await sync_to_async(lambda: self.async_client.force_login(
            User.objects.get_or_create(username='Me', email="me@example.com")[0]))()

        json_schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["greeting", "count"]
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce JSON output based on a schema."},
                {"role": "user", "content": "Generate JSON matching the provided schema."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": json_schema}
            },
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": True
        }
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        response = await self.async_client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=headers,
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        streamed_lines = await _read_streaming_response_lines(response)
        self.assertGreater(len(streamed_lines), 0, "Should receive at least one streamed data chunk.")

        content_pieces = _parse_streamed_content_pieces(streamed_lines)
        full_content = "".join(content_pieces).strip()
        result = json.loads(full_content)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(json_schema["properties"].keys()))
        self.assertIsInstance(result["greeting"], str)
        self.assertIsInstance(result["count"], int)

        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(len(requests), 1, "There should be exactly one request after schema generation streaming.")
        req = requests[0]
        self.assertIn("chat/completions", req.path,
                      "Request endpoint should be for chat completion (streaming schema generation).")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 (streaming schema generation)")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 (streaming schema generation)")


class ListModelsIntegrationTest(GatewayIntegrationTestCase):

    def _send_model_list_request(self):
        return self.client.get(
            f"/models",
            data='',
            content_type="application/json",
            headers=_build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        )

    def test_list_models(self):
        """
        Sends a request to list available models from the vLLM server using the Django test client.
        After the request, checks that the database contains one request and the endpoint matches.
        """
        response = self._send_model_list_request()

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        response_json = response.json()
        print(f"\nList models response: {response_json}")

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        print(f"Available model IDs: {model_ids}")
        self.assertIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")

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
        response = self.client.get(
            f"/models",
            data='',
            content_type="application/json",
            **headers
        )

        # Should be 401 Unauthorized or 403 Forbidden
        self.assertIn(
            response.status_code, [401, 403],
            f"Expected 401 or 403 for invalid token, got {response.status_code}: {response.content}"
        )

        # There should be no request recorded in the database (or possibly one, depending on implementation)
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

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}: {response.content}")

        response_json = response.json()
        print(f"\nList models response: {response_json}")

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        print(f"Available model IDs: {model_ids}")
        self.assertEqual(len(model_ids), len(model_list) - 1)
        self.assertNotIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")


class TokenLimitTest(ChatCompletionsIntegrationTest):
    def setup_limits(self, kind: str, field: str, value: int):
        """
        Set a rate limit for the given kind ('org', 'team', 'user') and field.
        Returns the object whose limit was set.

        For 'team', also ensure the service account is associated with the test token.
        """
        if kind == "org":
            org = Org.objects.get(name="E060")
            setattr(org, field, value)
            org.save(update_fields=[field])
            return org
        elif kind == "team":
            team = Team.objects.get(name="Whale")
            setattr(team, field, value)
            team.save(update_fields=[field])
            # Ensure a service account exists for the team and associate it with the token
            # Only associate the service account with the token if the token does not already have a service account

            service_account = ServiceAccount.objects.create(team=team, name="Whale Service")

            from management.models import Token
            token = Token.objects.filter(key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN)).first()
            if not token:
                raise RuntimeError("Could not find Token associated with AQUEDUCT_ACCESS_TOKEN.")

            # Only set the service_account if it is not already set
            if getattr(token, "service_account_id", None) is None:
                token.service_account = service_account
                token.save(update_fields=["service_account"])
            elif token.service_account_id != service_account.id:
                # If the token is already associated with a different service account, raise an error
                raise RuntimeError(
                    f"Token is already associated with a different service account (id={token.service_account_id})."
                )
            # Otherwise, already associated with the correct service account, do nothing
            return team
        elif kind == "user":
            user = User.objects.get(username="Me")
            profile = user.profile if hasattr(user, "profile") else UserProfile.objects.get(user=user)
            setattr(profile, field, value)
            profile.save(update_fields=[field])
            return profile
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def _rate_limit_test_template(self, kind: str, field: str, value: int, messages, max_tokens, limit_desc):
        """
        Generic template for rate limit tests for org, team, or user.
        Uses the Django test client to POST to the chat completion endpoint.
        """
        # Set the limit
        obj = self.setup_limits(kind, field, value)

        response1 = self._send_chat_completion(messages, max_tokens=max_tokens, temperature=0.0)
        self.assertEqual(
            response1.status_code, 200,
            f"Expected 200 OK, got {response1.status_code}: {response1.content}"
        )
        response_json = response1.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertGreater(len(chat_completion.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1,
                         f"There should be exactly one request after first chat completion ({limit_desc}).")
        req = requests[0]
        self.assertIn("chat/completions", req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        response2 = self._send_chat_completion(messages, max_tokens=max_tokens, temperature=0.0)
        self.assertEqual(
            response2.status_code, 429,
            f"Expected 429 Too Many Requests, got {response2.status_code}: {response2.content}"
        )
        # Optionally, check error message
        try:
            error_json = response2.json()
            self.assertTrue(
                "rate limit" in str(error_json).lower() or "429" in str(error_json),
                f"Expected rate limit error message, got: {error_json}"
            )
        except Exception:
            # If not JSON, just check content
            self.assertTrue(
                "rate limit" in response2.content.decode().lower() or "429" in response2.content.decode(),
                f"Expected rate limit error message, got: {response2.content}"
            )

    def test_org_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Org 'E060' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="org requests_per_minute"
        )

    def test_team_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of Team 'Whale' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="team requests_per_minute"
        )

    def test_user_rate_limit_requests_per_minute(self):
        """
        Edits the requests_per_minute of UserProfile for user 'Me' to 1, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="requests_per_minute",
            value=1,
            messages=self.MESSAGES,
            max_tokens=5,
            limit_desc="user requests_per_minute"
        )

    def test_org_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="org input_tokens_per_minute"
        )

    def test_team_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="team input_tokens_per_minute"
        )

    def test_user_rate_limit_input_tokens_per_minute(self):
        """
        Edits the input_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="input_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=1,
            limit_desc="user input_tokens_per_minute"
        )

    def test_org_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Org 'E060' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="org",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="org output_tokens_per_minute"
        )

    def test_team_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of Team 'Whale' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="team",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="team output_tokens_per_minute"
        )

    def test_user_rate_limit_output_tokens_per_minute(self):
        """
        Edits the output_tokens_per_minute of UserProfile for user 'Me' to 5, then makes two requests.
        The second request should raise a 429 HTTP error.
        """
        self._rate_limit_test_template(
            kind="user",
            field="output_tokens_per_minute",
            value=5,
            messages=self.MESSAGES,
            max_tokens=10,
            limit_desc="user output_tokens_per_minute"
        )
