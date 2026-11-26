"""Tests for streaming responses returned through the `log_request` decorator (#246).

Django only returns a request's DB connection to the pool on `request_finished`,
which does not run when an exception escapes a stream. `log_request` therefore
releases the connection before streaming and ends a failing stream quietly.
"""

import asyncio
import contextlib
import time
from typing import ClassVar
from unittest.mock import patch

from asgiref.sync import sync_to_async
from django.db import connections
from django.http import StreamingHttpResponse
from django.test import TransactionTestCase, override_settings
from fakeredis import FakeServer
from fakeredis.aioredis import FakeRedis

from gateway.decorators import check_token_concurrency_limits, log_request
from gateway.decorators.limits import RedisConcurrencyLimiter, token_concurrency_key
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

    @override_settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED=True)
    async def test_concurrency_slot_released_when_stream_ends(self) -> None:
        """`log_request` wraps the concurrency limit decorators, so its stream guard sits
        outside the lease's stream wrapper. The connection is still released before
        streaming, and the slot when the stream ends in any way."""
        redis = FakeRedis(server=FakeServer(), decode_responses=True)
        try:
            await self._check_slot_released_when_stream_ends(redis)
        finally:
            await redis.aclose()

    async def _check_slot_released_when_stream_ends(self, redis: FakeRedis) -> None:
        limiter = RedisConcurrencyLimiter(redis)
        self.token.concurrent_requests_limit = 1
        key = token_concurrency_key(self.token)

        for outcome in ("complete", "error", "cancel", "disconnect"):
            with self.subTest(outcome=outcome):
                self.events.clear()
                first_chunk_sent = asyncio.Event()

                async def stream(outcome=outcome, first_chunk_sent=first_chunk_sent):
                    self.events.append("chunk")
                    yield b"data: 1\n\n"
                    first_chunk_sent.set()
                    if outcome == "error":
                        raise RuntimeError("upstream died mid-stream")
                    if outcome in {"cancel", "disconnect"}:
                        await asyncio.sleep(3600)  # Block until cancelled or closed

                async def streaming_view(request, *args, stream=stream, **kwargs):
                    return StreamingHttpResponse(stream(), content_type="text/event-stream")

                with patch("gateway.decorators.limits.get_limiter", return_value=limiter):
                    response = await log_request(check_token_concurrency_limits(streaming_view))(
                        _make_request(),
                        token=self.token,
                        pydantic_model=None,
                        request_start=time.monotonic(),
                    )
                    self.assertEqual(await redis.zcard(key), 1)

                    logs = (
                        self.assertLogs("aqueduct", level="ERROR")
                        if outcome == "error"
                        else contextlib.nullcontext()
                    )
                    with logs:
                        if outcome == "disconnect":
                            # Like Django when the client goes away mid-stream: it closes
                            # `aiter(response)`, swallows the cancellation and then
                            # closes the response.
                            content = aiter(response)
                            await asyncio.wait_for(anext(content), timeout=1)
                            await content.aclose()
                            await sync_to_async(response.close)()
                        else:
                            task = asyncio.create_task(_consume(response.streaming_content))
                            await asyncio.wait_for(first_chunk_sent.wait(), timeout=1)
                            if outcome == "cancel":
                                task.cancel()
                            await asyncio.gather(task, return_exceptions=True)

                    # Closing the response starts the release in a task on the loop.
                    for _ in range(50):
                        if await redis.zcard(key) == 0:
                            break
                        await asyncio.sleep(0.01)
                    self.assertEqual(await redis.zcard(key), 0)
                    self.assertEqual(self.events[:2], ["close", "chunk"])


async def _consume(stream) -> list[bytes]:
    return [chunk async for chunk in stream]
