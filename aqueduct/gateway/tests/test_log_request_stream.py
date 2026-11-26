"""Tests for streaming responses returned through the `log_request` decorator (#246).

Django only returns a request's DB connection to the pool on `request_finished`,
which does not run when an exception escapes a stream. `log_request` therefore
releases the connection before streaming and ends a failing stream quietly.
"""

import asyncio
import time
from typing import ClassVar
from unittest.mock import patch

from django.db import connections
from django.http import StreamingHttpResponse
from django.test import TransactionTestCase

from gateway.decorators import log_request
from gateway.tests.test_log_request_cancel import _make_request
from management.models import Token


class LogRequestStreamTests(TransactionTestCase):
    # Connections are not closed inside a transaction, so this cannot be a `TestCase`.
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.token = Token.objects.get(name="My Token")
        self.events: list[str] = []
        wrapper_class = type(connections["default"])
        real_close = wrapper_class.close

        def spy_close(conn) -> None:
            self.events.append("close")
            real_close(conn)

        patcher = patch.object(wrapper_class, "close", autospec=True, side_effect=spy_close)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def _stream_through_log_request(self, stream) -> StreamingHttpResponse:
        async def streaming_view(request, *args, **kwargs):
            return StreamingHttpResponse(stream, content_type="text/event-stream")

        return await log_request(streaming_view)(
            _make_request(path="/mcp-servers/test/mcp"),
            token=self.token,
            pydantic_model=None,
            request_start=time.monotonic(),
        )

    async def test_connection_released_before_first_chunk(self) -> None:
        async def stream():
            self.events.append("chunk")
            yield b"data: 1\n\n"

        response = await self._stream_through_log_request(stream())
        chunks = [chunk async for chunk in response.streaming_content]

        self.assertEqual(chunks, [b"data: 1\n\n"])
        self.assertEqual(self.events, ["close", "chunk"])

    async def test_exception_mid_stream_ends_stream_quietly(self) -> None:
        async def stream():
            yield b"data: 1\n\n"
            raise RuntimeError("upstream died mid-stream")

        response = await self._stream_through_log_request(stream())
        with self.assertLogs("aqueduct", level="ERROR") as caught:
            chunks = [chunk async for chunk in response.streaming_content]

        self.assertEqual(chunks, [b"data: 1\n\n"])
        self.assertIn("Error while streaming response", "\n".join(caught.output))

    async def test_cancellation_still_propagates(self) -> None:
        first_chunk_sent = asyncio.Event()

        async def stream():
            yield b"data: 1\n\n"
            first_chunk_sent.set()
            await asyncio.sleep(3600)  # Block until cancelled

        response = await self._stream_through_log_request(stream())

        async def consume() -> None:
            async for _chunk in response.streaming_content:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first_chunk_sent.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
