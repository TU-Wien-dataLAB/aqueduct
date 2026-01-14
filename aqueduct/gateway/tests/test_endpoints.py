import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from asgiref.sync import async_to_sync, sync_to_async
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TransactionTestCase, override_settings
from django.urls import reverse
from httpx import Request as HttpxRequest
from httpx import Response
from litellm.types.utils import ModelResponse
from openai.types.audio import Transcription
from openai.types.chat import ChatCompletion
from openai.types.image import Image
from openai.types.images_response import ImagesResponse

from gateway.config import get_router_config
from gateway.tests.utils import (
    _build_chat_headers,
    _build_chat_payload,
    _parse_streamed_content_pieces,
    _read_streaming_response_lines,
)
from gateway.tests.utils.base import (
    INTEGRATION_TEST_BACKEND,
    ROUTER_CONFIG,
    GatewayIntegrationTestCase,
)
from gateway.tests.utils.mock_server import MockConfig, MockStreamingConfig
from management.models import Org, Request, ServiceAccount, Team, Token, UserProfile

User = get_user_model()


class EmbeddingTest(GatewayIntegrationTestCase):
    model = "Qwen-0.5B" if INTEGRATION_TEST_BACKEND == "vllm" else "text-embedding-ada-002"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.url = reverse("gateway:embeddings")

    @override_settings(RELAY_REQUEST_TIMEOUT=5)
    def test_embeddings(self):
        """
        Sends a simple embeddings request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... Requires GatewayIntegrationTestCase "
                "to manage multiple servers!"
            )

        assert self.model in ROUTER_CONFIG
        payload = {"model": self.model, "input": ["The quick brown fox jumps over the lazy dog."]}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        # print(f"\nEmbeddings response: {response_json}")

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
        self.assertEqual(req.user_id, "")


class ChatCompletionsBase(GatewayIntegrationTestCase):
    MESSAGES = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a short poem!"},
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.url = reverse("gateway:chat_completions")

    def _build_chat_completion_request(self, messages, stream=False, **payload_kwargs) -> dict:
        """
        Helper to build headers, payload, and endpoint for chat completion requests.
        """
        payload = _build_chat_payload(self.model, messages, stream=stream, **payload_kwargs)
        return dict(path=self.url, data=json.dumps(payload), headers=self.headers)

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

        with self.mock_server.patch_external_api():
            response = await self.async_client.post(**request, content_type="application/json")
        return response


class ChatCompletionsIntegrationTest(ChatCompletionsBase):
    def test_chat_completion(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """

        with self.mock_server.patch_external_api():
            response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        # Parse the response as JSON and convert to OpenAI ChatCompletion type for compatibility
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)

        # print(f"\nChat completion response: {chat_completion}")

        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertIsInstance(chat_completion.choices, list)
        self.assertGreater(len(chat_completion.choices), 0)

        first_choice = chat_completion.choices[0]
        self.assertTrue(hasattr(first_choice, "message"))
        self.assertIsNotNone(first_choice.message)
        self.assertTrue(hasattr(first_choice.message, "content"))
        self.assertIsNotNone(first_choice.message.content)

        # response_text = first_choice.message.content.strip()
        # print(response_text)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after chat completion."
        )
        req = requests[0]
        self.assertIn(self.url, req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")
        self.assertEqual(req.user_id, "")

    @override_settings(TIKA_SERVER_URL=None)
    def test_chat_completion_base64_file_input(self):
        """
        Sends a chat completion request with base64 encoded file input.
        Tests the file upload functionality using base64 encoding.
        """
        # Create a simple text file content and encode it as base64
        file_content = b"This is a test file content for base64 encoding."
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "test.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with (
            patch(
                "gateway.views.decorators.extract_text_with_tika",
                return_value="This is a test file content for base64 encoding.",
            ),
            self.mock_server.patch_external_api(),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content
        self.assertIsNotNone(content)
        content = content.strip()
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after base64 file input."
        )
        req = requests[0]
        self.assertIn(
            self.url, req.path, "Request endpoint should be chat completion for base64 file input."
        )
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for base64 file input")
        self.assertGreater(
            req.output_tokens, 0, "output_tokens should be > 0 for base64 file input"
        )

    def test_chat_completion_file_id_input(self):
        """
        Sends a chat completion request with file_id input.
        Tests the file upload functionality using a file ID from the files API.
        """
        # First, upload a file using the files API
        file_content = b'{"custom_id": "test_file_id_input"}\n'
        upload_file = SimpleUploadedFile(
            "test_file_id.jsonl", file_content, content_type="application/jsonl"
        )

        # Prepare headers for file upload (remove Content-Type for multipart)
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers.pop("Content-Type", None)

        # Upload the file
        with self.mock_server.patch_external_api():
            upload_response = self.client.post(
                "/files", {"file": upload_file, "purpose": "user_data"}, headers=headers
            )
        self.assertEqual(
            upload_response.status_code, 200, f"File upload failed: {upload_response.json()}"
        )
        upload_data = upload_response.json()
        file_id = upload_data["id"]

        # Now use the file_id in a chat completion request
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {"type": "file", "file": {"file_id": file_id}},
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with (
            patch(
                "gateway.views.decorators.extract_text_with_tika",
                return_value="This is a test file content for base64 encoding.",
            ),
            self.mock_server.patch_external_api(),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests),
            2,
            "There should be exactly two requests after file_id input (file upload + chat completion).",
        )

        # Get the chat completion request (should be the one with /chat/completions path)
        chat_request = next(r for r in requests if self.url in r.path)
        req = chat_request
        self.assertIn(
            self.url, req.path, "Request endpoint should be chat completion for file_id input."
        )
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for file_id input")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 for file_id input")

    def test_chat_completion_file_id_not_found(self):
        """
        Sends a chat completion request with a non-existent file_id.
        Should raise a 404 error.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {"type": "file", "file": {"file_id": "non-existent-file-id"}},
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            404,
            f"Expected 404 Not Found, got {response.status_code}: {response.content}",
        )

    def test_chat_completion_file_id_different_user(self):
        """
        Sends a chat completion request with a file_id that was created by a different user.
        Should raise a 404 error.
        """
        # First, upload a file using the files API with the default user
        file_content = b'{"custom_id": "test_file_id_input"}\n'
        upload_file = SimpleUploadedFile(
            "test_file_id.jsonl", file_content, content_type="application/jsonl"
        )

        # Prepare headers for file upload (remove Content-Type for multipart)
        headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
        headers.pop("Content-Type", None)

        # Upload the file
        with self.mock_server.patch_external_api():
            upload_response = self.client.post(
                "/files", {"file": upload_file, "purpose": "user_data"}, headers=headers
            )
        self.assertEqual(
            upload_response.status_code, 200, f"File upload failed: {upload_response.json()}"
        )
        upload_data = upload_response.json()
        file_id = upload_data["id"]

        UPDATED_ACCESS_TOKEN, _ = self.create_new_user()
        headers = _build_chat_headers(UPDATED_ACCESS_TOKEN)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {"type": "file", "file": {"file_id": file_id}},
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with (
            patch(
                "gateway.views.decorators.extract_text_with_tika",
                return_value="This is a test file content for base64 encoding.",
            ),
            self.mock_server.patch_external_api(),
        ):
            response = self.client.post(
                self.url, data=json.dumps(payload), headers=headers, content_type="application/json"
            )

        self.assertEqual(
            response.status_code,
            404,
            f"Expected 404 Not Found, got {response.status_code}: {response.content}",
        )

    @override_settings(AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB=1)
    def test_chat_completion_base64_file_size_limit_individual(self):
        """
        Tests that individual files uploaded via base64 are rejected if their size exceeds
        `AQUEDUCT_CHAT_COMPLETIONS_MAX_FILE_SIZE_MB`.
        """
        # Create a file content that exceeds 1MB
        file_size = 1024 * 1024 + 1  # 1MB + 1 byte
        file_content = b"x" * file_size
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "large_file.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400 Bad Request for file size > 1MB, got {response.status_code}: {response.content}",
        )
        self.assertIn(b"File too large", response.content)

    @override_settings(AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB=1)
    def test_chat_completion_base64_file_size_limit_total(self):
        """
        Tests that files upload via base64 is rejected if their total size exceeds
        `AQUEDUCT_CHAT_COMPLETIONS_MAX_TOTAL_SIZE_MB`.
        """
        # Create multiple files that together exceed 1MB
        # Each file is 350 kB, so 3 files ~1 MB total
        individual_file_size = 350 * 1024  # 350 kB each
        file_content = b"x" * individual_file_size
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in these files?"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "large_file_1.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": "large_file_2.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                        {
                            "type": "file",
                            "file": {
                                "filename": "large_file_3.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with (
            patch(
                "gateway.views.decorators.extract_text_with_tika",
                return_value="This is a test file content for base64 encoding.",
            ),
            self.mock_server.patch_external_api(),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400 Bad Request for total file size > 1MB, got {response.status_code}: "
            f"{response.content}",
        )
        self.assertIn(b"Files too large in total", response.content)

    def test_tika_error_response(self):
        """Handle errors from tika requests (extract text from file) gracefully."""
        # Create a simple text file content and encode it as base64
        file_content = b"This is a test file content for base64 encoding."
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this file?"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "test.txt",
                                "file_data": f"data:text/plain;base64,{file_base64}",
                            },
                        },
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }
        tika_response_mock = Response(
            request=HttpxRequest(method="PUT", url="http://example.com/tika"),
            json={"error": "Mocked Tika server error"},
            status_code=500,
        )

        with (
            patch(
                "gateway.views.decorators.httpx.AsyncClient.put", return_value=tika_response_mock
            ),
            self.mock_server.patch_external_api(),
        ):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400, got {response.status_code}: {response.content}",
        )
        self.assertIn("Tika error extracting text from file", response.json()["error"])

    @override_settings(RELAY_REQUEST_TIMEOUT=0.0001)
    def test_chat_completion_timeout(self):
        """
        Sends a simple chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        with self.mock_server.patch_external_api():
            response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(
            response.status_code,
            504,
            f"Expected 504 Gateway Timeout, got {response.status_code}: {response.content}",
        )

    @override_settings(RELAY_REQUEST_TIMEOUT=0.0001)
    def test_chat_completion_timeout_is_logged(self):
        """
        Test that timeout requests ARE logged to the database.

        This test verifies the bug fix where timeout requests were not appearing
        in the usage dashboard because they weren't being saved to the Request table.

        Bug scenario:
        - User makes request that times out
        - Request should still be logged with status_code=504
        - Request should appear in usage dashboard
        """
        # Clear any existing requests
        Request.objects.all().delete()

        with self.mock_server.patch_external_api():
            response = self._send_chat_completion(self.MESSAGES)

        # Verify we get a timeout response
        self.assertEqual(
            response.status_code,
            504,
            f"Expected 504 Gateway Timeout, got {response.status_code}: {response.content}",
        )

        # THIS IS THE CRITICAL CHECK: Verify request was logged despite timeout
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests),
            1,
            f"Expected 1 request to be logged even on timeout, but found {len(requests)}. "
            "Timeout requests MUST be logged for usage tracking!",
        )

        # Verify the logged request has correct information
        req = requests[0]
        self.assertEqual(
            req.status_code,
            504,
            f"Logged streaming timeout request should have status_code=504, got {req.status_code}",
        )
        self.assertIn(
            "chat/completions",
            req.path,
            f"Request path should be /chat/completions, got {req.path}",
        )
        self.assertIsNotNone(
            req.response_time_ms, "Response time should be recorded even for streaming timeout"
        )
        self.assertGreater(
            req.response_time_ms,
            0,
            f"Response time should be > 0 for streaming timeout, got {req.response_time_ms}",
        )
        # Access token_id instead of token to avoid async issues
        self.assertIsNotNone(req.token_id, "Token should be recorded for streaming timeout")
        self.assertEqual(
            req.model,
            self.model,
            f"Model should be {self.model} for streaming timeout, got {req.model}",
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
        await sync_to_async(
            lambda: self.async_client.force_login(
                User.objects.get_or_create(username="Me", email="me@example.com")[0]
            )
        )()

        with self.mock_server.patch_external_api():
            response = await self._send_chat_completion_streaming(self.MESSAGES)

        # Should be a StreamingHttpResponse with status 200
        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")

        # Collect all streamed lines (each line is a data: ... event)
        streamed_lines = await _read_streaming_response_lines(response)

        self.assertGreater(
            len(streamed_lines), 0, "Should receive at least one streamed data chunk."
        )

        # Parse each chunk as JSON and collect content pieces
        content_pieces = _parse_streamed_content_pieces(streamed_lines)
        full_content = "".join(content_pieces).strip()
        # print(f"Full streamed content: {full_content}")
        self.assertTrue(full_content, "Streamed content should not be empty.")

        # Check that the database contains one request and endpoint matches
        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after streaming chat completion."
        )
        req = requests[0]
        self.assertIn(
            self.url, req.path, "Request endpoint should be for chat completion (streaming)."
        )
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 (streaming)")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0 (streaming)")

    @override_settings(RELAY_REQUEST_TIMEOUT=0.0001)
    @async_to_sync
    async def test_chat_completion_streaming_relay_request_timeout(self):
        """
        Sends a streaming chat completion request to the vLLM server using the Django test client.
        After the request, checks that the database contains one request,
        the endpoint matches, and input/output tokens are > 0.
        """
        # For some reason authentication does not work in async test case...
        await sync_to_async(
            lambda: self.async_client.force_login(
                User.objects.get_or_create(username="Me", email="me@example.com")[0]
            )
        )()

        with self.mock_server.patch_external_api():
            response = await self._send_chat_completion_streaming(self.MESSAGES)

        self.assertEqual(
            response.status_code, 504, f"Expected 504 Gateway Timeout, got {response.status_code}"
        )

    @override_settings(RELAY_REQUEST_TIMEOUT=0.0001)
    @async_to_sync
    async def test_chat_completion_streaming_timeout_is_logged(self):
        """
        Test that streaming timeout requests are also logged to the database.

        This test verifies that even when streaming requests time out,
        they are still logged to the Request table for usage tracking.

        Bug scenario:
        - User makes streaming request that times out
        - Request should still be logged with status_code=504
        - Request should appear in usage dashboard
        """
        # Clear existing requests
        await sync_to_async(Request.objects.all().delete)()

        # Authenticate
        await sync_to_async(
            lambda: self.async_client.force_login(
                User.objects.get_or_create(username="Me", email="me@example.com")[0]
            )
        )()

        with self.mock_server.patch_external_api():
            response = await self._send_chat_completion_streaming(self.MESSAGES)

        # Verify we get a timeout response
        self.assertEqual(
            response.status_code,
            504,
            f"Expected 504 Gateway Timeout for streaming, got {response.status_code}",
        )

        # THIS IS THE CRITICAL CHECK: Verify streaming timeout request was logged
        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertGreater(
            len(requests),
            0,
            f"Expected at least 1 request to be logged for streaming timeout, but found {len(requests)}. "
            "Streaming timeout requests MUST be logged for usage tracking!",
        )

        # Verify the logged request has correct information
        req = requests[0]
        self.assertEqual(
            req.status_code,
            504,
            f"Logged streaming timeout request should have status_code=504, got {req.status_code}",
        )
        self.assertIn(
            "chat/completions",
            req.path,
            f"Request path should be /chat/completions, got {req.path}",
        )
        self.assertIsNotNone(
            req.response_time_ms, "Response time should be recorded even for streaming timeout"
        )
        self.assertGreater(
            req.response_time_ms,
            0,
            f"Response time should be > 0 for streaming timeout, got {req.response_time_ms}",
        )
        # Access token_id instead of token to avoid async issues
        self.assertIsNotNone(req.token_id, "Token should be recorded for streaming timeout")
        self.assertEqual(
            req.model,
            self.model,
            f"Model should be {self.model} for streaming timeout, got {req.model}",
        )

    def test_chat_completion_excluded_model(self):
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()
        assert len(org.excluded_models) == 1

        with self.mock_server.patch_external_api():
            response = self._send_chat_completion(self.MESSAGES)

        self.assertEqual(
            response.status_code,
            404,
            f"Expected 404 Model not found, got {response.status_code}: {response.content}",
        )

    def test_chat_completion_unknown_model(self):
        payload = {"model": "unknown-model", "messages": self.MESSAGES}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )
        self.assertEqual(
            response.status_code,
            400,
            f"Expected 400 Bad Request, got {response.status_code}: {response.content}",
        )
        self.assertIn("There is no 'model_name' with this string", response.json()["error"])

    @override_settings(RELAY_REQUEST_TIMEOUT=5)
    def test_chat_completion_schema_generation(self):
        """
        Sends a chat completion request with a JSON schema, non-streaming.
        Verifies the response content adheres to the schema and logs the request.
        """
        json_schema = {
            "type": "object",
            "properties": {"greeting": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["greeting", "count"],
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce JSON output based on a schema."},
                {"role": "user", "content": "Generate JSON matching the provided schema."},
            ],
            # { "type": "json_schema", "json_schema": {...} }
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": json_schema},
            },
            "max_completion_tokens": 50,
        }

        expected = MockConfig(
            response_data={
                "id": "chatcmpl-123456789",
                "created": 1768397207,
                "model": "gpt-4.1-nano-2025-04-14",
                "object": "chat.completion",
                "system_fingerprint": "fp_f0bc439dc3",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": '{"greeting":"Hello, world!","count":1}',
                            "role": "assistant",
                        },
                    }
                ],
                "usage": {
                    "completion_tokens": 13,
                    "prompt_tokens": 59,
                    "total_tokens": 72,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
                },
            }
        )
        with self.mock_server.patch_external_api("chat/completions", expected):
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        result = json.loads(content)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(json_schema["properties"].keys()))
        self.assertIsInstance(result["greeting"], str)
        self.assertIsInstance(result["count"], int)

        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after schema generation."
        )
        req = requests[0]
        self.assertIn(
            self.url, req.path, "Request endpoint should be for chat completion schema generation."
        )
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0 for schema generation")
        self.assertGreater(
            req.output_tokens, 0, "output_tokens should be > 0 for schema generation"
        )

    def test_chat_completion_multimodal_input(self):
        if INTEGRATION_TEST_BACKEND == "vllm":
            self.skipTest(
                "Tests not adapted for vLLM yet... Requires GatewayIntegrationTestCase "
                "to manage multiple servers!"
            )

        with open(
            Path(__file__).parent / "resources" / "Polytechnisches-Institut-1823.jpg", "rb"
        ) as image_file:
            img_b64 = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What’s in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ],
            "max_completion_tokens": 50,
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )
        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        content = chat_completion.choices[0].message.content.strip()
        self.assertGreater(len(content), 0)

        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after multimodal input."
        )
        req = requests[0]
        self.assertIn(
            self.url, req.path, "Request endpoint should be chat completion for multimodal input."
        )
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
        await sync_to_async(
            lambda: self.async_client.force_login(
                User.objects.get_or_create(username="Me", email="me@example.com")[0]
            )
        )()

        json_schema = {
            "type": "object",
            "properties": {"greeting": {"type": "string"}, "count": {"type": "integer"}},
            "required": ["greeting", "count"],
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You produce JSON output based on a schema."},
                {"role": "user", "content": "Generate JSON matching the provided schema."},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": json_schema},
            },
            "max_completion_tokens": 50,
            "stream": True,
        }

        expected_data = [
            b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"{\\"greeting\\":","role":"assistant"}}],"stream_options":{"include_usage":true}}\n\n',
            b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\\"Hello, world!\\","}}],"stream_options":{"include_usage":true}}\n\n',
            b'data: {"id":"chatcmpl-12345","created":1768398242,"model":"gpt-4.1-nano","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"\\"count\\":1}"}}],"stream_options":{"include_usage":true}}\n\n',
        ]
        expected = MockStreamingConfig(response_data=expected_data)

        with self.mock_server.patch_external_api("chat/completions", expected):
            response = await self.async_client.post(
                self.url,
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200, f"Expected 200 OK, got {response.status_code}")
        streamed_lines = await _read_streaming_response_lines(response)
        self.assertGreater(
            len(streamed_lines), 0, "Should receive at least one streamed data chunk."
        )

        content_pieces = _parse_streamed_content_pieces(streamed_lines)
        full_content = "".join(content_pieces).strip()
        result = json.loads(full_content)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(json_schema["properties"].keys()))
        self.assertIsInstance(result["greeting"], str)
        self.assertIsInstance(result["count"], int)

        requests = await sync_to_async(lambda: list(Request.objects.all()))()
        self.assertEqual(
            len(requests),
            1,
            "There should be exactly one request after schema generation streaming.",
        )
        req = requests[0]
        self.assertIn(
            self.url,
            req.path,
            "Request endpoint should be for chat completion (streaming schema generation).",
        )
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(
            req.input_tokens, 0, "input_tokens should be > 0 (streaming schema generation)"
        )
        self.assertGreater(
            req.output_tokens, 0, "output_tokens should be > 0 (streaming schema generation)"
        )


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
        # print(f"\nList models response: {response_json}")

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        # print(f"Available model IDs: {model_ids}")
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
        response = self.client.get("/models", content_type="application/json", headers=headers)

        # Should be 401 Unauthorized or 403 Forbidden
        self.assertIn(
            response.status_code,
            [401, 403],
            f"Expected 401 or 403 for invalid token, got {response.status_code}: {response.content}",
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

        self.assertEqual(
            response.status_code,
            200,
            f"Expected 200 OK, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        # print(f"\nList models response: {response_json}")

        # OpenAI API returns an object with a 'data' attribute that is a list of models
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)

        # Check that at least one model matches the expected model name
        model_ids = [m["id"] for m in response_json["data"] if "id" in m]
        # print(f"Available model IDs: {model_ids}")
        self.assertEqual(len(model_ids), len(model_list) - 1)
        self.assertNotIn(self.model, model_ids)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(len(requests), 1, "There should be exactly one request after list models.")
        req = requests[0]
        self.assertIn("models", req.path, "Request endpoint should be for model listing.")


class TokenLimitTest(ChatCompletionsBase):
    def _setup_limits(self, kind: str, field: str, value: int):
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

            token = Token.objects.filter(
                key_hash=Token._hash_key(self.AQUEDUCT_ACCESS_TOKEN)
            ).first()
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
            profile = (
                user.profile if hasattr(user, "profile") else UserProfile.objects.get(user=user)
            )
            setattr(profile, field, value)
            profile.save(update_fields=[field])
            return profile
        else:
            raise ValueError(f"Unknown kind: {kind}")

    def _rate_limit_test_template(
        self, kind: str, field: str, value: int, messages, max_completion_tokens, limit_desc
    ):
        """
        Generic template for rate limit tests for org, team, or user.
        Uses the Django test client to POST to the chat completion endpoint.
        """
        # Set the limit
        self._setup_limits(kind, field, value)

        with self.mock_server.patch_external_api():
            response1 = self._send_chat_completion(
                messages, max_completion_tokens=max_completion_tokens
            )
        self.assertEqual(
            response1.status_code,
            200,
            f"Expected 200 OK, got {response1.status_code}: {response1.content}",
        )
        response_json = response1.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)
        self.assertGreater(len(chat_completion.choices), 0)

        # Check that the database contains one request and endpoint matches
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests),
            1,
            f"There should be exactly one request after first chat completion ({limit_desc}).",
        )
        req = requests[0]
        self.assertIn(self.url, req.path, "Request endpoint should be for chat completion.")
        self.assertIsNotNone(req.input_tokens)
        self.assertIsNotNone(req.output_tokens)
        self.assertGreater(req.input_tokens, 0, "input_tokens should be > 0")
        self.assertGreater(req.output_tokens, 0, "output_tokens should be > 0")

        # Second request should fail with 429
        with self.mock_server.patch_external_api():
            response2 = self._send_chat_completion(
                messages, max_completion_tokens=max_completion_tokens
            )
        self.assertEqual(
            response2.status_code,
            429,
            f"Expected 429 Too Many Requests, got {response2.status_code}: {response2.content}",
        )
        # Optionally, check error message
        try:
            error_json = response2.json()
            self.assertTrue(
                "rate limit" in str(error_json).lower() or "429" in str(error_json),
                f"Expected rate limit error message, got: {error_json}",
            )
        except Exception:
            # If not JSON, just check content
            self.assertTrue(
                "rate limit" in response2.content.decode().lower()
                or "429" in response2.content.decode(),
                f"Expected rate limit error message, got: {response2.content}",
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
            max_completion_tokens=5,
            limit_desc="org requests_per_minute",
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
            max_completion_tokens=5,
            limit_desc="team requests_per_minute",
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
            max_completion_tokens=5,
            limit_desc="user requests_per_minute",
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
            max_completion_tokens=1,
            limit_desc="org input_tokens_per_minute",
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
            max_completion_tokens=1,
            limit_desc="team input_tokens_per_minute",
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
            max_completion_tokens=1,
            limit_desc="user input_tokens_per_minute",
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
            max_completion_tokens=10,
            limit_desc="org output_tokens_per_minute",
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
            max_completion_tokens=10,
            limit_desc="team output_tokens_per_minute",
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
            max_completion_tokens=10,
            limit_desc="user output_tokens_per_minute",
        )


class ModelAliasConfigValidationTest(TransactionTestCase):
    """
    Tests for model alias configuration validation.
    These tests validate that the router config correctly validates aliases when loaded.
    """

    def test_config_load_with_valid_aliases(self):
        """
        Test that config loads successfully when aliases are unique and valid.
        """
        # Mock config with unique aliases
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main", "coding"]},
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"mode": "embedding", "aliases": ["embedding"]},
                },
            ]
        }

        with patch("builtins.open"):
            with patch("yaml.safe_load", return_value=mock_config) as mock_load:
                get_router_config.cache_clear()

                loaded_config = get_router_config()

                # Config should load without errors
                self.assertIsInstance(loaded_config, dict)
                self.assertIn("model_list", loaded_config)

                # Verify aliases are in the config
                first_model = loaded_config["model_list"][0]
                self.assertIn("model_info", first_model)
                self.assertIn("aliases", first_model["model_info"])
                self.assertEqual(first_model["model_info"]["aliases"], ["main", "coding"])
                mock_load.assert_called_once()

    def test_config_load_with_duplicate_aliases_raises_error(self):
        """
        Test that config validation raises error when aliases are not unique.
        """
        # Mock config with duplicate "main" alias
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},  # Duplicate!
                },
            ]
        }

        with patch("builtins.open"):
            with patch("yaml.safe_load", return_value=mock_config) as mock_load:
                get_router_config.cache_clear()

                # Should raise RuntimeError due to duplicate aliases
                with self.assertRaises(RuntimeError) as context:
                    get_router_config()

                # Verify the error message mentions the duplicate alias
                self.assertIn("Duplicate alias", str(context.exception))
                self.assertIn("main", str(context.exception))
                mock_load.assert_called_once()

    def test_config_load_with_multiple_aliases_per_model(self):
        """
        Test that a single model can have multiple aliases.
        """
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main", "coding", "default", "primary"]},
                }
            ]
        }

        with patch("builtins.open"):
            with patch("yaml.safe_load", return_value=mock_config) as mock_load:
                get_router_config.cache_clear()

                loaded_config = get_router_config()

                # Config should load without errors
                self.assertIsInstance(loaded_config, dict)
                model = loaded_config["model_list"][0]
                self.assertEqual(len(model["model_info"]["aliases"]), 4)
                self.assertIn("main", model["model_info"]["aliases"])
                self.assertIn("primary", model["model_info"]["aliases"])
                mock_load.assert_called_once()

    def test_alias_case_sensitivity(self):
        """
        Test that aliases are case-sensitive and "Main" and "main" are distinct.
        """
        # Config with case-sensitive aliases on different models
        mock_config = {
            "model_list": [
                {
                    "model_name": "gpt-4.1-nano",
                    "litellm_params": {
                        "model": "openai/gpt-4.1-nano",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["Main"]},  # Capital M
                },
                {
                    "model_name": "text-embedding-ada-002",
                    "litellm_params": {
                        "model": "openai/text-embedding-ada-002",
                        "api_key": "os.environ/OPENAI_API_KEY",
                    },
                    "model_info": {"aliases": ["main"]},  # Lowercase m
                },
            ]
        }

        with patch("builtins.open"):
            with patch("yaml.safe_load", return_value=mock_config) as mock_load:
                get_router_config.cache_clear()

                loaded_config = get_router_config()

                # Should load successfully since "Main" and "main" are different
                self.assertIsInstance(loaded_config, dict)

                # Collect all aliases
                all_aliases = []
                for model in loaded_config["model_list"]:
                    if "model_info" in model and "aliases" in model["model_info"]:
                        all_aliases.extend(model["model_info"]["aliases"])

                # Both should be present and distinct
                self.assertIn("Main", all_aliases)
                self.assertIn("main", all_aliases)
                self.assertEqual(len([a for a in all_aliases if a.lower() == "main"]), 2)
                mock_load.assert_called_once()


class ModelAliasRoutingTest(GatewayIntegrationTestCase):
    """
    Tests for model alias resolution during API requests.
    These tests check that aliases are correctly resolved to model names.
    """

    mock_response = ModelResponse()
    tts_model = "gpt-4o-mini-tts"
    stt_model = "whisper-1"
    image_gen_model = "dall-e-2"

    def setUp(self):
        super().setUp()
        get_router_config.cache_clear()

    def test_chat_completion_with_alias(self):
        """
        Test that chat completion requests using an alias are routed correctly.
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello!"},
        ]

        # Try to use alias instead of model name
        payload = {"model": "main", "messages": messages, "max_tokens": 10}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working. "
            f"Expected 200, got {response.status_code}: {response.content}",
        )

        response_json = response.json()
        chat_completion = ChatCompletion.model_validate(response_json)
        self.assertIsNotNone(chat_completion)
        self.assertTrue(chat_completion.choices)

    def test_speech_endpoint_with_alias(self):
        """Test that TTS endpoint correctly resolves model aliases."""
        payload = {
            "model": "tts",  # Using alias instead of actual model name
            "input": "Testing alias resolution for text-to-speech.",
            "voice": "alloy",
            "response_format": "mp3",
        }

        with self.mock_server.patch_external_api():
            response = self.client.post(
                "/audio/speech",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for TTS. Expected 200, got {response.status_code}",
        )

        # Check that the database contains one request
        requests = list(Request.objects.all())
        self.assertEqual(
            len(requests), 1, "There should be exactly one request after speech generation."
        )

    def test_transcriptions_endpoint_with_alias(self):
        """Test that STT endpoint correctly resolves model aliases."""
        # Create a simple audio file for testing (using a small mock file)
        test_audio_content = b"mock audio data"
        test_audio_file = SimpleUploadedFile(
            "test.mp3", test_audio_content, content_type="audio/mp3"
        )

        with patch("gateway.views.utils.get_openai_client") as mock_client:
            mock_openai_client = AsyncMock()
            mock_openai_client.audio.transcriptions.create.return_value = Transcription(
                text="How much is the fish?"
            )
            mock_client.return_value = mock_openai_client

            headers = _build_chat_headers(self.AQUEDUCT_ACCESS_TOKEN)
            # Remove Content-Type for multipart form data
            headers.pop("Content-Type", None)

            with self.mock_server.patch_external_api():
                response = self.client.post(
                    "/audio/transcriptions",
                    {
                        "file": test_audio_file,
                        "model": "stt",
                    },  # Using alias instead of actual model name
                    headers=headers,
                )

            # Should return 200 with alias resolution
            self.assertEqual(
                response.status_code,
                200,
                f"Alias resolution not working for STT. "
                f"Expected 200, got {response.status_code}: {response.content}",
            )

            response_json = response.json()
            self.assertIn("text", response_json, "Response should contain 'text' field")

            # Check that the database contains one request
            requests = list(Request.objects.all())
            self.assertEqual(
                len(requests), 1, "There should be exactly one request after transcription."
            )

    def test_chat_completion_with_nonexistent_alias(self):
        """
        Test that requests with non-existent aliases return appropriate errors.
        """
        messages = [{"role": "user", "content": "Hello!"}]
        payload = {"model": "nonexistent-alias-12345", "messages": messages, "max_tokens": 10}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 400)

    def test_embeddings_with_alias(self):
        """
        Test that embedding requests using an alias are routed correctly.
        """
        payload = {"model": "embedding", "input": ["The quick brown fox."]}

        with self.mock_server.patch_external_api():
            response = self.client.post(
                "/embeddings",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Should return 200 with alias resolution
        self.assertEqual(
            response.status_code,
            200,
            f"Alias resolution not working for embeddings. Expected 200, got {response.status_code}",
        )

        response_json = response.json()
        self.assertIn("data", response_json)
        self.assertIsInstance(response_json["data"], list)
        self.assertGreater(len(response_json["data"]), 0)

    def test_image_generation_with_alias(self):
        """Test that image generation endpoint correctly resolves model aliases."""
        mock_image_object = Image(
            b64_json="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            revised_prompt="A beautiful landscape with mountains and a lake",
        )
        mock_image_response = ImagesResponse(data=[mock_image_object], created=123456789)

        with patch("gateway.views.utils.get_openai_client") as mock_client:
            mock_openai_client = AsyncMock()
            mock_openai_client.images.generate.return_value = mock_image_response

            mock_client.return_value = mock_openai_client

            payload = {
                "model": "image",  # Using alias instead of actual model name
                "prompt": "A beautiful landscape with mountains and a lake",
                "size": "256x256",
            }

            with self.mock_server.patch_external_api():
                response = self.client.post(
                    "/images/generations",
                    data=json.dumps(payload),
                    headers=self.headers,
                    content_type="application/json",
                )

            # Should return 200 with alias resolution
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

    def test_multiple_aliases_same_model(self):
        """
        Test that multiple aliases for the same model all work correctly.
        """
        messages = [{"role": "user", "content": "Test"}]
        aliases_to_test = ["main", "coding"]

        first_status = None
        for alias in aliases_to_test:
            payload = {"model": alias, "messages": messages, "max_tokens": 5}

            with self.mock_server.patch_external_api():
                response = self.client.post(
                    "/chat/completions",
                    data=json.dumps(payload),
                    headers=self.headers,
                    content_type="application/json",
                )

            # All should behave the same (either all work or all fail)
            if first_status is None:
                # Remember result for first alias
                first_status = response.status_code
            else:
                # Others should match
                self.assertEqual(
                    response.status_code,
                    first_status,
                    f"Alias '{alias}' returned {response.status_code}, "
                    f"but first alias returned {first_status}",
                )

    def test_excluded_model_alias_rejected(self):
        """
        Test that aliases for excluded models are rejected.
        """
        # Exclude the main model for the organization
        org = Org.objects.get(name="E060")
        org.add_excluded_model(self.model)
        org.save()

        messages = [{"role": "user", "content": "Test"}]

        # Test with actual model name
        payload = {"model": self.model, "messages": messages, "max_tokens": 5}

        with self.mock_server.patch_external_api():
            response_actual = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Test with alias
        payload = {"model": "main", "messages": messages, "max_tokens": 5}

        with self.mock_server.patch_external_api():
            response_alias = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Both should be rejected (404)
        self.assertEqual(
            response_actual.status_code,
            404,
            f"Expected 404 for excluded model, got {response_actual.status_code}",
        )

        # Alias should also be rejected since it resolves to excluded model
        self.assertEqual(
            response_alias.status_code,
            404,
            f"Alias to excluded model should return 404, got {response_alias.status_code}",
        )

    def test_alias_case_sensitivity_in_requests(self):
        """
        Test that alias case sensitivity is preserved in actual requests.
        """
        messages = [{"role": "user", "content": "Test"}]

        # Test with correct case
        payload = {"model": "main", "messages": messages, "max_tokens": 5}

        with self.mock_server.patch_external_api():
            response_lowercase = self.client.post(
                "/chat/completions",
                data=json.dumps(payload),
                headers=self.headers,
                content_type="application/json",
            )

        # Should return 200 with alias resolution
        self.assertEqual(
            response_lowercase.status_code,
            200,
            f"Alias resolution not working for lowercase. Expected 200, got {response_lowercase.status_code}",
        )

        # Test with wrong case
        payload = {"model": "Main", "messages": messages, "max_tokens": 5}
        response_uppercase = self.client.post(
            "/chat/completions",
            data=json.dumps(payload),
            headers=self.headers,
            content_type="application/json",
        )

        # "Main" should fail since only "main" is defined (case-sensitive)
        self.assertIn(
            response_uppercase.status_code,
            [400, 404],
            f"Wrong case alias 'Main' should fail, got {response_uppercase.status_code}",
        )
