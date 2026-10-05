"""Tests for non-streaming client-disconnect handling in ``log_request``.

When a client disconnects while the view is still awaiting the upstream LLM
(before any response has been returned), Django's ASGI handler cancels the
request task, which raises ``asyncio.CancelledError`` inside the view.
``log_request`` catches that and records status 499 on the request log.

This asserts the non-streaming behaviour, complementing the streaming tests
in ``test_stream_status.py`` (which exercise ``_openai_stream`` directly).
"""

import asyncio
import time
from types import SimpleNamespace
from typing import ClassVar

from asgiref.sync import async_to_sync
from django.test import TestCase

from gateway.views.decorators import log_request
from management.models import Request, Token


def _make_request(path: str = "/v1/chat/completions") -> SimpleNamespace:
    """Minimal stand-in for an ASGI request, exposing only what log_request reads."""
    return SimpleNamespace(
        path=path,
        method="POST",
        headers=SimpleNamespace(get=lambda key, default="": default),
        META=SimpleNamespace(get=lambda key, default="": default),
    )


class LogRequestCancelTests(TestCase):
    fixtures: ClassVar[list[str]] = ["gateway_data.json"]

    def setUp(self):
        self.token = Token.objects.get(name="My Token")

    def test_non_streaming_client_disconnect_records_499(self):
        """A disconnect while awaiting the upstream is recorded as 499."""
        started = asyncio.Event()

        async def hanging_view(request, *args, **kwargs):
            started.set()
            await asyncio.sleep(3600)  # simulate awaiting the upstream LLM

        wrapped = log_request(hanging_view)

        async def run():
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
