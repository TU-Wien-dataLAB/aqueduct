"""Tests for the streaming status-code recording in the request log.

Covers the mapping the stream generator writes to ``Request.status_code``:

- Clean completion  -> 200
- Client closes     -> 499 (client-closed-request)
- Upstream failure  -> 500
"""

import json

from asgiref.sync import async_to_sync
from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase

from gateway.views.utils import _openai_stream
from management.models import Org, Request, Token, UserGroup, UserProfile

User = get_user_model()


class _Chunk:
    """Minimal stand-in for a streamed chunk that only needs model_dump_json."""

    def model_dump_json(self, *args, **kwargs) -> str:
        return json.dumps({"choices": [{"delta": {"content": "hi"}}]})


class OpenAIStreamStatusTests(TestCase):
    def setUp(self):
        self.org = Org.objects.create(name="stream-org")
        self.user = User.objects.create_user(username="streamuser", email="stream@example.com")
        UserProfile.objects.create(user=self.user, org=self.org)
        Group.objects.get_or_create(name=UserGroup.USER.value)
        self.token = Token(name="stream-token", user=self.user)
        self.token._set_new_key()
        self.token.save()

    def _request_log(self) -> Request:
        request_log = Request(token=self.token, model="gpt-4.1-nano")
        request_log.save()
        return request_log

    def test_clean_completion_records_200(self):
        async def fake_stream():
            yield _Chunk()
            yield _Chunk()

        request_log = self._request_log()
        stream = _openai_stream(fake_stream(), request_log)

        async def consume_all():
            async for _ in stream:
                pass

        async_to_sync(consume_all)()
        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 200)

    def test_client_close_records_499(self):
        """When the client closes the connection mid-stream we record 499."""

        async def fake_stream():
            yield _Chunk()
            yield _Chunk()
            yield _Chunk()

        request_log = self._request_log()
        stream = _openai_stream(fake_stream(), request_log)

        async def consume_then_disconnect():
            it = stream.__aiter__()
            await it.__anext__()
            await it.__anext__()
            await stream.aclose()

        async_to_sync(consume_then_disconnect)()
        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 499)

    def test_upstream_failure_records_500(self):
        """An upstream error part-way through the stream is recorded as 500."""

        async def failing_stream():
            yield _Chunk()
            raise RuntimeError("upstream exploded")

        request_log = self._request_log()
        stream = _openai_stream(failing_stream(), request_log)

        async def consume_until_failure():
            async for _ in stream:
                pass

        with self.assertRaises(RuntimeError):
            async_to_sync(consume_until_failure)()

        request_log.refresh_from_db()
        self.assertEqual(request_log.status_code, 500)
