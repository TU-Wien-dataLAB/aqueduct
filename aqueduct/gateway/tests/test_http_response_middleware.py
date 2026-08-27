import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from django.contrib.auth import get_user_model
from django.core.handlers.asgi import ASGIRequest
from django.http import JsonResponse, StreamingHttpResponse
from django.test import AsyncRequestFactory, override_settings
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import Usage as LitellmUsage

from gateway.middleware import HttpResponseMiddleware
from gateway.tests.utils.base import GatewayBatchesTestCase
from gateway.views.utils import RawJsonResponse, RawStreamingResponse
from management.models import Token, Usage

User = get_user_model()

ROOT_DIR = Path(__file__).parent.parent.parent.parent


@override_settings(
    OIDC_OP_JWKS_ENDPOINT="https://example.com/application/o/example/jwks/",
    LITELLM_ROUTER_CONFIG_FILE_PATH=Path(ROOT_DIR / "example_router_config.yaml"),
    AQUEDUCT_FILES_API_URL="https://files-api.example.com",
    AQUEDUCT_FILES_API_KEY="test_key",
)
class TestHttpResponseMiddleware(GatewayBatchesTestCase):
    """Test that HttpResponseMiddleware properly transforms raw responses from views."""

    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.factory = AsyncRequestFactory()
        cls.token_header = {"Authorization": "Bearer sk-123abc"}
        cls.token = Token.objects.first()

    def test_transforms_raw_json_response(self):
        """RawJsonResponse should become JsonResponse with merged headers."""
        raw_response = RawJsonResponse(
            data={"key": "value"}, headers={"X-Custom": "test"}, status=200
        )

        def get_response(req: ASGIRequest):
            return raw_response

        middleware = HttpResponseMiddleware(get_response)
        request = self.factory.get("/test", **self.token_header)
        response = middleware(request)

        self.assertIsInstance(response, JsonResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["X-Custom"], "test")
        self.assertEqual(response["Content-Type"], "application/json")
        self.assertEqual(json.loads(response.content), {"key": "value"})

    async def test_transforms_raw_streaming_response_mcp(self):
        """RawStreamingResponse for MCP paths should become StreamingHttpResponse."""

        async def streaming_content():
            yield {"result": "mcp-data 1"}
            yield {"result": "mcp-data 2"}

        def transform(chunk: dict[str, Any]):
            result = chunk["result"]
            if isinstance(result, str):
                chunk["result"] = result + " custom suffix"
            return chunk

        raw_response = RawStreamingResponse(
            streaming_content=streaming_content(),
            request_log=None,
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            transforms=[transform],
        )

        def get_response(req: ASGIRequest):
            raw_response.headers["MCP-session-id"] = "mcp-test-id"
            return raw_response

        middleware = HttpResponseMiddleware(get_response)
        request = self.factory.get("/mcp-servers/test", **self.token_header)
        response = middleware(request)

        self.assertIsInstance(response, StreamingHttpResponse)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(response["mcp-session-id"], "mcp-test-id")

        async for chunk in response.streaming_content:
            chunk_text = chunk.decode("utf-8")
            self.assertTrue(chunk_text.startswith("data: "))
            content = json.loads(chunk_text.removeprefix("data: "))
            self.assertTrue(content["result"].endswith(" custom suffix"))

    def test_passes_through_regular_response(self):
        """Non-raw responses should pass through unchanged."""
        regular_response = JsonResponse({"status": "ok"})

        def get_response(req: ASGIRequest):
            return regular_response

        middleware = HttpResponseMiddleware(get_response)
        request = self.factory.get("/test")
        response = middleware(request)

        self.assertIs(response, regular_response)

    async def test_openai_stream_logs_token_usage(self):
        """OpenAI streaming should update request_log and apply transforms."""

        async def streaming_content():
            yield ModelResponseStream(
                id="chatcmpl-stream-reasoning",
                created=1768398242,
                model="gpt-4.1-nano",
                object="chat.completion.chunk",
                stream=True,
                stream_options={"include_usage": True},
                choices=[{"index": 0, "delta": {"role": "assistant"}}],
                usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        def add_reasoning(chunk: ModelResponseStream) -> ModelResponseStream:
            choices = chunk.get("choices", [])
            for choice in choices:
                message = choice.get("delta")
                if message:
                    message["reasoning"] = "Deep chain of thought..."
            return chunk

        mock_request_log = AsyncMock(id=1, token_usage=None, response_time_ms=None)
        mock_request_log.asave = AsyncMock()

        raw_response = RawStreamingResponse(
            streaming_content=streaming_content(),
            request_log=mock_request_log,
            transforms=[add_reasoning],
        )

        def get_response(req: ASGIRequest):
            return raw_response

        middleware = HttpResponseMiddleware(get_response)
        request = self.factory.post("/v1/chat/completions", **self.token_header)
        response = middleware(request)

        async for chunk in response.streaming_content:
            chunk_text = chunk.decode("utf-8")
            self.assertTrue(chunk_text.startswith("data: "))
            if chunk_text != "data: [DONE]\n\n":
                content = json.loads(chunk_text.removeprefix("data: "))
                choices = content["choices"]
                for choice in choices:
                    self.assertEqual(choice["delta"]["reasoning"], "Deep chain of thought...")

        # The last "DONE" chunk should be added to an openai stream:
        self.assertEqual(chunk_text, "data: [DONE]\n\n")
        mock_request_log.asave.assert_called_once()
        self.assertEqual(mock_request_log.token_usage, Usage(input_tokens=10, output_tokens=5))
        self.assertIsNotNone(mock_request_log.response_time_ms)
