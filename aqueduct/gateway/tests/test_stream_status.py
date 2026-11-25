"""Tests for status-code recording in `RawStreamingResponse` streaming.

Streaming requests only learn their final status when the stream ends:
- clean completion -> 200,
- client disconnect (generator close / task cancellation) -> 499,
- upstream failure -> 500 (the client gets an error event instead of [DONE]).
"""

import asyncio
import json
from typing import ClassVar
from unittest.mock import patch

from django.db import connections
from django.test import TestCase, TransactionTestCase
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import Usage as LitellmUsage

from gateway.response_type import RawStreamingResponse
from management.models import Request, Token, Usage


def _chunk(text: str = "hi", usage: LitellmUsage | None = None) -> ModelResponseStream:
    kwargs = {}
    if usage is not None:
        kwargs["usage"] = usage
    return ModelResponseStream(
        id="chatcmpl-stream-status",
        created=1768398242,
        model="gpt-4.1-nano",
        object="chat.completion.chunk",
        stream=True,
        choices=[{"index": 0, "delta": {"content": text}}],
        **kwargs,
    )


def _underlying_stream(response: RawStreamingResponse):
    """Return the underlying `_iter_stream` async generator of a response.

    Django's `streaming_content` property wraps the generator in a fresh
    generator on every access; closing that wrapper does not close
    `_iter_stream` itself. The tests need to close/cancel `_iter_stream`
    directly to simulate a client disconnect.
    """
    return response._iterator


class StreamingStatusTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.token = Token.objects.get(name="My Token")

    async def _make_request_log(self) -> Request:
        return await Request.objects.acreate(
            token=self.token, model="gpt-4.1-nano", method="POST", path="/v1/chat/completions"
        )

    async def test_clean_completion_records_200(self) -> None:
        async def fake_stream():
            yield _chunk("Hel")
            yield _chunk("lo")
            yield _chunk(usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

        request_log = await self._make_request_log()
        response = RawStreamingResponse(streaming_content=fake_stream(), request_log=request_log)

        chunks = [chunk async for chunk in _underlying_stream(response)]

        # OpenAI streams end with the "data:\n" terminator
        self.assertEqual(chunks.pop(), b"data: [DONE]\n\n")
        self.assertEqual(len(chunks), 3)

        await request_log.arefresh_from_db()
        self.assertEqual(request_log.status_code, 200)
        self.assertIsNotNone(request_log.response_time_ms)
        self.assertEqual(request_log.token_usage, Usage(input_tokens=10, output_tokens=5))

    async def test_client_close_mid_stream_records_499(self) -> None:
        async def fake_stream():
            yield _chunk()
            yield _chunk()
            yield _chunk()

        request_log = await self._make_request_log()
        response = RawStreamingResponse(streaming_content=fake_stream(), request_log=request_log)

        # Consume part of the stream, then close it like a disconnecting client would
        stream = _underlying_stream(response)
        async for _chunk_raw in stream:
            break
        await stream.aclose()

        await request_log.arefresh_from_db()
        self.assertEqual(request_log.status_code, 499)
        self.assertIsNotNone(request_log.response_time_ms)

    async def test_cancelled_stream_records_499(self) -> None:
        async def slow_stream():
            yield _chunk()
            await asyncio.sleep(3600)  # Block until cancelled

        request_log = await self._make_request_log()
        response = RawStreamingResponse(streaming_content=slow_stream(), request_log=request_log)

        task = asyncio.create_task(_consume(_underlying_stream(response)))
        # Give the task time to consume the first chunk and suspend in sleep()
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        await request_log.arefresh_from_db()
        self.assertEqual(request_log.status_code, 499)

    async def test_upstream_failure_emits_error_event_and_records_500(self) -> None:
        async def failing_stream():
            yield _chunk()
            raise RuntimeError("upstream exploded")

        request_log = await self._make_request_log()
        response = RawStreamingResponse(streaming_content=failing_stream(), request_log=request_log)

        with self.assertLogs("aqueduct", level="ERROR"):
            chunks = [chunk async for chunk in _underlying_stream(response)]

        # The stream ends with an error event instead of raising, and without [DONE]
        self.assertEqual(len(chunks), 2)
        self.assertNotIn(b"data: [DONE]\n\n", chunks)
        self.assertTrue(chunks[-1].startswith(b"data: "))
        self.assertTrue(chunks[-1].endswith(b"\n\n"))
        error = json.loads(chunks[-1].removeprefix(b"data: "))
        self.assertEqual(
            error,
            {"error": {"message": "Upstream stream failed", "type": "server_error", "code": 500}},
        )
        # The exception text must not reach the client
        self.assertNotIn(b"exploded", chunks[-1])

        await request_log.arefresh_from_db()
        self.assertEqual(request_log.status_code, 500)
        self.assertIsNotNone(request_log.response_time_ms)


class StreamingConnectionReleaseTests(TransactionTestCase):
    """Streams return the DB connection after each status record (#246).

    Saving the status checks a pooled connection out again; it must be closed in
    the same step, so that none is held when the stream ends, fails or is
    cancelled. Connections are not closed inside a transaction, so these tests
    cannot run in a `TestCase`.
    """

    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        token = Token.objects.get(name="My Token")
        self.request_log = Request.objects.create(
            token=token, model="gpt-4.1-nano", method="POST", path="/v1/chat/completions"
        )
        # The status code stored in the database each time a connection is closed
        self.status_at_close: list[int | None] = []
        wrapper_class = type(connections["default"])
        real_close = wrapper_class.close

        def spy_close(conn) -> None:
            status = Request.objects.get(pk=self.request_log.pk).status_code
            self.status_at_close.append(status)
            real_close(conn)

        patcher = patch.object(wrapper_class, "close", autospec=True, side_effect=spy_close)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_completion_releases_connection_after_recording_200(self) -> None:
        async def fake_stream():
            yield _chunk()

        response = RawStreamingResponse(
            streaming_content=fake_stream(), request_log=self.request_log
        )
        await _consume(_underlying_stream(response))

        self.assertEqual(self.status_at_close, [200])

    async def test_failure_releases_connection_after_recording_500(self) -> None:
        async def failing_stream():
            yield _chunk()
            raise RuntimeError("upstream exploded")

        response = RawStreamingResponse(
            streaming_content=failing_stream(), request_log=self.request_log
        )
        with self.assertLogs("aqueduct", level="ERROR"):
            await _consume(_underlying_stream(response))

        self.assertEqual(self.status_at_close, [500])

    async def test_client_close_releases_connection_after_recording_499(self) -> None:
        async def fake_stream():
            yield _chunk()
            yield _chunk()

        response = RawStreamingResponse(
            streaming_content=fake_stream(), request_log=self.request_log
        )
        stream = _underlying_stream(response)
        async for _chunk_raw in stream:
            break
        await stream.aclose()

        self.assertEqual(self.status_at_close, [499])

    async def test_cancel_releases_connection_after_recording_499(self) -> None:
        async def slow_stream():
            yield _chunk()
            await asyncio.sleep(3600)  # Block until cancelled

        response = RawStreamingResponse(
            streaming_content=slow_stream(), request_log=self.request_log
        )
        task = asyncio.create_task(_consume(_underlying_stream(response)))
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(self.status_at_close, [499])


async def _consume(stream) -> None:
    async for _ in stream:
        pass
