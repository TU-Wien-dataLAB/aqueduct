"""Tests for status-code recording in `RawStreamingResponse` streaming.

Streaming requests only learn their final status when the stream ends:
- clean completion -> 200,
- client disconnect (generator close / task cancellation) -> 499,
- upstream failure -> 500 (the client gets an error event instead of [DONE]).
"""

import asyncio
import json
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import litellm
from asgiref.sync import sync_to_async
from django.core.cache import cache
from django.db import connections
from django.test import TestCase, TransactionTestCase
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponseStream
from litellm.types.utils import Usage as LitellmUsage
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel

from gateway.rate_limiting import get_per_token_usage
from gateway.response_type import RawStreamingResponse
from gateway.tests.utils.base import ChatCompletionsBase
from management.models import Org, Request, Token, Usage


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


class _TextCompletionChunk(BaseModel):
    id: str = "cmpl-stream-status"
    model: str = "gpt-4.1-nano"
    choices: list[dict[str, Any]]


class AbortedAndFailedStreamTokenUsageTests(TestCase):
    """Tests that aborted (499) and failed (500) streams estimate and record token usage (#256)."""

    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        cache.clear()
        self.token = Token.objects.get(name="My Token")

    def tearDown(self) -> None:
        cache.clear()
        super().tearDown()

    async def _make_request_log(self, path: str = "/v1/chat/completions") -> Request:
        return await Request.objects.acreate(
            token=self.token, model="gpt-4.1-nano", method="POST", path=path
        )

    async def test_client_close_mid_stream_estimates_and_records_token_usage(self) -> None:
        messages = [{"role": "user", "content": "Explain rate limiting in one sentence."}]
        request_body: dict[str, object] = {"model": "gpt-4.1-nano", "messages": messages}

        async def fake_stream():
            yield _chunk("Rate limiting")
            yield _chunk(" caps token")
            yield _chunk(" consumption per window that is never forwarded")

        request_log = await self._make_request_log()
        response = RawStreamingResponse(
            streaming_content=fake_stream(), request_log=request_log, request_body=request_body
        )

        stream = _underlying_stream(response)
        await anext(stream)
        await anext(stream)
        await stream.aclose()

        await request_log.arefresh_from_db()
        expected_in = litellm.token_counter(model="gpt-4.1-nano", messages=messages)
        expected_out = litellm.token_counter(model="gpt-4.1-nano", text="Rate limiting caps token")
        self.assertEqual(request_log.status_code, 499)
        self.assertEqual(request_log.input_tokens, expected_in)
        self.assertEqual(request_log.output_tokens, expected_out)
        self.assertGreater(expected_in, 0)
        self.assertGreater(expected_out, 0)

        buckets = await sync_to_async(get_per_token_usage)([self.token.id])
        self.assertEqual(buckets[self.token.id]["min"]["in"], expected_in)
        self.assertEqual(buckets[self.token.id]["min"]["out"], expected_out)

    async def test_cancelled_stream_estimates_reasoning_without_double_counting(self) -> None:
        messages = [{"role": "user", "content": "What is 6 times 7?"}]
        request_body: dict[str, object] = {"model": "gpt-4.1-nano", "messages": messages}
        yielded_both = asyncio.Event()

        async def slow_stream():
            reasoning_chunk = ModelResponseStream(
                id="chatcmpl-reasoning",
                created=1768398242,
                model="gpt-4.1-nano",
                object="chat.completion.chunk",
                stream=True,
                choices=[
                    {
                        "index": 0,
                        "delta": {
                            "reasoning": "Six multiplied by seven is forty-two.",
                            "reasoning_content": "Six multiplied by seven is forty-two.",
                        },
                    }
                ],
            )
            yield reasoning_chunk
            yield _chunk("The answer is 42.")
            yielded_both.set()
            await asyncio.sleep(3600)

        request_log = await self._make_request_log()
        response = RawStreamingResponse(
            streaming_content=slow_stream(), request_log=request_log, request_body=request_body
        )

        task = asyncio.create_task(_consume(_underlying_stream(response)))
        await asyncio.wait_for(yielded_both.wait(), timeout=2)
        await asyncio.sleep(0.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        await request_log.arefresh_from_db()
        expected_in = litellm.token_counter(model="gpt-4.1-nano", messages=messages)
        expected_reasoning = litellm.token_counter(
            model="gpt-4.1-nano", text="Six multiplied by seven is forty-two."
        )
        expected_visible = litellm.token_counter(model="gpt-4.1-nano", text="The answer is 42.")
        expected_out = expected_reasoning + expected_visible

        self.assertEqual(request_log.status_code, 499)
        self.assertEqual(request_log.input_tokens, expected_in)
        self.assertEqual(request_log.reasoning_tokens, expected_reasoning)
        self.assertEqual(request_log.output_tokens, expected_out)

        buckets = await sync_to_async(get_per_token_usage)([self.token.id])
        self.assertEqual(buckets[self.token.id]["min"]["in"], expected_in)
        self.assertEqual(buckets[self.token.id]["min"]["out"], expected_out)

    async def test_upstream_failure_estimates_and_records_token_usage(self) -> None:
        messages = [{"role": "user", "content": "Write a short paragraph."}]
        request_body: dict[str, object] = {"model": "gpt-4.1-nano", "messages": messages}

        async def failing_stream():
            yield _chunk("Partial streamed answer before error")
            raise RuntimeError("upstream exploded")

        request_log = await self._make_request_log()
        response = RawStreamingResponse(
            streaming_content=failing_stream(), request_log=request_log, request_body=request_body
        )

        with self.assertLogs("aqueduct", level="ERROR"):
            _ = [chunk async for chunk in _underlying_stream(response)]

        await request_log.arefresh_from_db()
        expected_in = litellm.token_counter(model="gpt-4.1-nano", messages=messages)
        expected_out = litellm.token_counter(
            model="gpt-4.1-nano", text="Partial streamed answer before error"
        )
        self.assertEqual(request_log.status_code, 500)
        self.assertEqual(request_log.input_tokens, expected_in)
        self.assertEqual(request_log.output_tokens, expected_out)

        buckets = await sync_to_async(get_per_token_usage)([self.token.id])
        self.assertEqual(buckets[self.token.id]["min"]["in"], expected_in)
        self.assertEqual(buckets[self.token.id]["min"]["out"], expected_out)

    async def test_aborted_stream_preserves_reported_upstream_usage_when_present(self) -> None:
        messages = [{"role": "user", "content": "Short prompt"}]
        request_body: dict[str, object] = {"model": "gpt-4.1-nano", "messages": messages}

        async def partial_usage_then_fail():
            yield _chunk(
                "",
                usage=LitellmUsage(
                    prompt_tokens=37,
                    completion_tokens=0,
                    total_tokens=37,
                    prompt_tokens_details={"cached_tokens": 20},
                ),
            )
            yield _chunk("Generated text before crash")
            raise RuntimeError("upstream reset")

        request_log = await self._make_request_log()
        response = RawStreamingResponse(
            streaming_content=partial_usage_then_fail(),
            request_log=request_log,
            request_body=request_body,
        )

        with self.assertLogs("aqueduct", level="ERROR"):
            _ = [chunk async for chunk in _underlying_stream(response)]

        await request_log.arefresh_from_db()
        expected_out = litellm.token_counter(
            model="gpt-4.1-nano", text="Generated text before crash"
        )
        expected_prompt_tokens = 37
        expected_cached_tokens = 20
        self.assertEqual(request_log.status_code, 500)
        self.assertEqual(request_log.input_tokens, expected_prompt_tokens)
        self.assertEqual(request_log.cached_input_tokens, expected_cached_tokens)
        self.assertEqual(request_log.output_tokens, expected_out)

    async def test_responses_and_completions_aborted_streams_estimate_tokens(self) -> None:
        async def responses_stream():
            yield ResponseTextDeltaEvent(
                content_index=0,
                delta="Hello from responses stream",
                item_id="msg_1",
                logprobs=[],
                output_index=0,
                sequence_number=1,
                type="response.output_text.delta",
            )
            await asyncio.sleep(3600)

        req_resp = await self._make_request_log(path="/v1/responses")
        resp_obj = RawStreamingResponse(
            streaming_content=responses_stream(),
            request_log=req_resp,
            request_body={
                "model": "gpt-4.1-nano",
                "input": "Say hello",
                "instructions": "Be polite.",
            },
        )
        it = _underlying_stream(resp_obj)
        await anext(it)
        await it.aclose()
        await req_resp.arefresh_from_db()
        self.assertEqual(req_resp.status_code, 499)
        self.assertGreater(req_resp.input_tokens, 0)
        self.assertGreater(req_resp.output_tokens, 0)

        async def completions_stream():
            yield _TextCompletionChunk(choices=[{"index": 0, "text": "Once upon a time in Vienna"}])
            raise RuntimeError("upstream failed")

        req_cmpl = await self._make_request_log(path="/v1/completions")
        cmpl_obj = RawStreamingResponse(
            streaming_content=completions_stream(),
            request_log=req_cmpl,
            request_body={"model": "gpt-4.1-nano", "prompt": "Start a story:"},
        )
        with self.assertLogs("aqueduct", level="ERROR"):
            _ = [c async for c in _underlying_stream(cmpl_obj)]
        await req_cmpl.arefresh_from_db()
        self.assertEqual(req_cmpl.status_code, 500)
        self.assertGreater(req_cmpl.input_tokens, 0)
        self.assertGreater(req_cmpl.output_tokens, 0)


class AbortedStreamRateLimitBypassTests(ChatCompletionsBase):
    """End-to-end check that incomplete streams count toward token rate limits (#256)."""

    def setUp(self) -> None:
        super().setUp()
        cache.clear()

    def tearDown(self) -> None:
        cache.clear()
        super().tearDown()

    async def test_aborted_chat_stream_enforces_output_token_rate_limit(self) -> None:
        output_limit = 3

        def set_org_limit() -> None:
            org = Org.objects.get(name="E060")
            org.output_tokens_per_minute = output_limit
            org.save(update_fields=["output_tokens_per_minute"])

        await sync_to_async(set_org_limit)()
        first_chunk_yielded = asyncio.Event()

        async def fake_acompletion(*args, **kwargs):
            async def gen():
                yield _chunk("One two three four five six seven eight")
                first_chunk_yielded.set()
                await asyncio.sleep(3600)

            return CustomStreamWrapper(
                completion_stream=gen(), model="gpt-4.1-nano", logging_obj=MagicMock()
            )

        with patch("gateway.views.chat_completions.get_router") as mock_get_router:
            mock_get_router.return_value.acompletion = fake_acompletion
            req = self._build_chat_completion_request(self.MESSAGES, stream=True)
            response = await self.async_client.post(**req, content_type="application/json")
            self.assertEqual(response.status_code, 200)

            task = asyncio.create_task(_consume(response.streaming_content))
            await asyncio.wait_for(first_chunk_yielded.wait(), timeout=2)
            await asyncio.sleep(0.02)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

            logged = await Request.objects.alatest("id")
            self.assertEqual(logged.status_code, 499)
            self.assertGreater(logged.input_tokens, 0)
            self.assertGreater(logged.output_tokens, output_limit)

            second = await self.async_client.post(**req, content_type="application/json")
            self.assertEqual(second.status_code, 429)

    async def test_failed_chat_stream_enforces_input_token_rate_limit(self) -> None:
        input_limit = 5

        def set_org_limit() -> None:
            org = Org.objects.get(name="E060")
            org.input_tokens_per_minute = input_limit
            org.save(update_fields=["input_tokens_per_minute"])

        await sync_to_async(set_org_limit)()

        async def fake_acompletion(*args, **kwargs):
            async def gen():
                yield _chunk("Partial")
                raise RuntimeError("upstream stream died")

            return CustomStreamWrapper(
                completion_stream=gen(), model="gpt-4.1-nano", logging_obj=MagicMock()
            )

        with patch("gateway.views.chat_completions.get_router") as mock_get_router:
            mock_get_router.return_value.acompletion = fake_acompletion
            req = self._build_chat_completion_request(self.MESSAGES, stream=True)
            response = await self.async_client.post(**req, content_type="application/json")
            self.assertEqual(response.status_code, 200)

            with self.assertLogs("aqueduct", level="ERROR"):
                await _consume(response.streaming_content)

            logged = await Request.objects.alatest("id")
            self.assertEqual(logged.status_code, 500)
            self.assertGreater(logged.input_tokens, input_limit)
            self.assertGreater(logged.output_tokens, 0)

            second = await self.async_client.post(**req, content_type="application/json")
            self.assertEqual(second.status_code, 429)
