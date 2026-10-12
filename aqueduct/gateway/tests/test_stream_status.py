import json
from typing import ClassVar

from asgiref.sync import async_to_sync
from django.test import TestCase

from gateway.response_type import RawStreamingResponse
from management.models import Request, Token


class _Chunk:
    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"choices": [{"delta": {"content": "hi"}}]})


class OpenAIStreamStatusTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.token = Token.objects.get(name="My Token")

    def _request_log(self) -> Request:
        request_log = Request(token=self.token, model="gpt-4.1-nano")
        request_log.save()
        return request_log

    def _openai_stream(self, content, request_log: Request):
        response = RawStreamingResponse(streaming_content=content, request_log=request_log)
        return response.streaming_content

    def test_clean_completion_records_200(self) -> None:
        async def fake_stream():
            yield _Chunk()
            yield _Chunk()
            yield _Chunk()

        request_log = self._request_log()
        stream = self._openai_stream(fake_stream(), request_log)

        async def consume_all() -> None:
            async for _ in stream:
                pass

        async_to_sync(consume_all)()

        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 200)

    def test_client_close_records_499(self) -> None:
        async def fake_stream():
            yield _Chunk()
            yield _Chunk()
            yield _Chunk()

        request_log = self._request_log()
        stream = self._openai_stream(fake_stream(), request_log)

        async def consume_then_disconnect() -> None:
            it = stream.__aiter__()
            await it.__anext__()
            await it.__anext__()
            await stream.aclose()

        async_to_sync(consume_then_disconnect)()

        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 499)

    def test_upstream_failure_records_500(self) -> None:
        async def failing_stream():
            yield _Chunk()
            raise RuntimeError("upstream exploded")

        request_log = self._request_log()
        stream = self._openai_stream(failing_stream(), request_log)

        async def consume_until_failure() -> None:
            async for _ in stream:
                pass

        with self.assertRaises(RuntimeError):
            async_to_sync(consume_until_failure)()

        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 500)
