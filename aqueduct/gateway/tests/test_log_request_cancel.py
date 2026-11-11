"""Tests for the 499 client-disconnect handling in the `log_request` decorator."""

import asyncio
import time
from types import SimpleNamespace
from typing import ClassVar

from django.test import TestCase

from gateway.decorators import log_request
from management.models import Request, Token


def _make_request(path: str = "/v1/chat/completions") -> SimpleNamespace:
    """A minimal request object with the attributes accessed by `log_request`."""
    return SimpleNamespace(
        path=path,
        method="POST",
        headers=SimpleNamespace(get=lambda name, default="": default),
        META=SimpleNamespace(get=lambda name, default=None: default),
    )


class LogRequestCancelledTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self) -> None:
        self.token = Token.objects.get(name="My Token")

    async def test_cancelled_view_records_499_and_reraises(self) -> None:
        started = asyncio.Event()

        async def hanging_view(request, *args, **kwargs):
            started.set()
            await asyncio.sleep(3600)  # Block until the client disconnects

        wrapped = log_request(hanging_view)

        task = asyncio.create_task(
            wrapped(
                _make_request(),
                token=self.token,
                pydantic_model={"model": "gpt-4.1-nano"},
                request_start=time.monotonic(),
            )
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        request_log = await Request.objects.aget(token=self.token)
        self.assertEqual(request_log.status_code, 499)
        self.assertEqual(request_log.path, "/v1/chat/completions")
        self.assertEqual(request_log.model, "gpt-4.1-nano")
        self.assertIsNotNone(request_log.processing_time_ms)
        self.assertIsNotNone(request_log.response_time_ms)
