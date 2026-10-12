import asyncio
import time
from types import SimpleNamespace
from typing import ClassVar

from asgiref.sync import async_to_sync
from django.test import TestCase

from gateway.decorators import log_request
from management.models import Request, Token


def _make_request(path: str = "/v1/chat/completions") -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        method="POST",
        headers=SimpleNamespace(get=lambda default: ""),
        META=SimpleNamespace(get=lambda default: ""),
    )


class LogRequestCancelTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.token = Token.objects.get(name="My Token")

    def test_non_streaming_client_disconnect_records_499(self) -> None:
        started = asyncio.Event()

        async def hanging_view(request, *args, **kwargs):
            started.set()
            await asyncio.sleep(3600)

        wrapped = log_request(hanging_view)

        async def run() -> None:
            task = asyncio.create_task(
                wrapped(_make_request(), token=self.token, request_start=time.monotonic())
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        async_to_sync(run)()

        request_log = Request.objects.get(token=self.token)
        self.assertEqual(request_log.status_code, 499)
        self.assertIsNotNone(request_log.response_time_ms)
        self.assertIsNotNone(request_log.processing_time_ms)
        self.assertEqual(request_log.path, "/v1/chat/completions")
