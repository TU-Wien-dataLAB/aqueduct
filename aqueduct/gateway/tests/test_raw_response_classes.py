import json
from unittest.mock import AsyncMock, patch

from django.test import SimpleTestCase
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import Usage as LitellmUsage
from mcp import JSONRPCResponse
from mcp.types import JSONRPCMessage
from pydantic import BaseModel

from gateway.views.utils import RawJsonResponse, RawStreamingResponse
from management.models import Usage


class TestRawJsonResponse(SimpleTestCase):
    def test_content_is_serialized_lazily_and_cached(self):
        response = RawJsonResponse(data={"key": "value"})
        # Nothing is serialized before `.content` is accessed for the first time.
        self.assertIsNone(response._content)

        original_dump = response._dump_data
        with patch.object(response, "_dump_data", side_effect=original_dump) as dump:
            first = response.content
            second = response.content
            third = response.content

        # Accessing `.content` multiple times must not re-serialize the data.
        dump.assert_called_once()
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(json.loads(first), {"key": "value"})

    def test_data_serialized_to_json(self):
        class Nested(BaseModel):
            foo: str

        data = {"name": "test", "nested": Nested(foo="bar"), "items": [Nested(foo="baz")]}
        response = RawJsonResponse(data=data)
        self.assertEqual(
            json.loads(response.content),
            {"name": "test", "nested": {"foo": "bar"}, "items": [{"foo": "baz"}]},
        )

    def test_rejects_non_dict_data(self):
        with self.assertRaises(TypeError):
            RawJsonResponse(data="not a dict")  # type: ignore[arg-type]


class TestRawStreamingResponse(SimpleTestCase):
    async def test_openai_mode_applies_transforms_and_logs_request(self):
        async def streaming_content():
            yield ModelResponseStream(
                id="chatcmpl-stream-reasoning",
                created=1768398242,
                model="gpt-4.1-nano",
                object="chat.completion.chunk",
                stream=True,
                stream_options={"include_usage": True},
                choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}],
                usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        def add_reasoning(chunk: ModelResponseStream) -> ModelResponseStream:
            choices = chunk.get("choices", [])
            for choice in choices:
                message = choice.get("delta")
                if message:
                    message["reasoning"] = "Deep chain of thought..."
            return chunk

        mock_request_log = AsyncMock(token_usage=None, response_time_ms=None)
        mock_request_log.asave = AsyncMock()

        response = RawStreamingResponse(
            streaming_content=streaming_content(),
            request_log=mock_request_log,
            transforms=[add_reasoning],
        )

        chunks = [chunk async for chunk in response.streaming_content]
        self.assertEqual(chunks.pop(), b"data: [DONE]\n\n")
        self.assertEqual(len(chunks), 1)

        payload = json.loads(chunks[0].decode().removeprefix("data: "))
        self.assertEqual(payload["choices"][0]["delta"]["reasoning"], "Deep chain of thought...")

        mock_request_log.asave.assert_called_once()
        self.assertIsNotNone(mock_request_log.response_time_ms)
        self.assertEqual(mock_request_log.token_usage, Usage(input_tokens=10, output_tokens=5))

    async def test_mcp_mode_applies_transforms_without_request_log(self):
        async def streaming_content():
            yield JSONRPCResponse(jsonrpc="2.0", id=0, result={"test": "hello"})
            yield JSONRPCResponse(jsonrpc="2.0", id=0, result={"test": "world"})

        def append_suffix(chunk: JSONRPCMessage) -> JSONRPCMessage:
            chunk.result["test"] = chunk.result["test"] + "!"
            return chunk

        response = RawStreamingResponse(
            streaming_content=streaming_content(),
            request_log=None,
            transforms=[append_suffix],
            mode="mcp",
        )

        chunks = [chunk async for chunk in response.streaming_content]

        # MCP responses do not get the OpenAI "data: [DONE]" terminator.
        self.assertEqual(len(chunks), 2)
        for chunk_raw in chunks:
            payload = json.loads(chunk_raw.decode().removeprefix("data: "))
            self.assertTrue(payload["result"]["test"].endswith("!"))
