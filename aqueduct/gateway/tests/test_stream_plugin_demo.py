"""Demo: a streaming plugin against POST /chat/completions.

Run with:
    cd aqueduct && uv run python manage.py test gateway.tests.test_stream_plugin_demo -v 2

It seeds a plugin, sends a streaming chat completion through the mock server,
and prints what the plugin's hooks observe.
"""

# ruff: noqa: T201  # this demo deliberately prints an interactive transcript
from http import HTTPStatus
from typing import ClassVar

from django.test import override_settings

from gateway.tests.test_endpoints import ChatCompletionsBase
from gateway.tests.utils import _read_streaming_response_lines
from management.models import Snippet, SnippetType
from management.plugins import _plugin_class

STREAMED: list[str] = []
CALLED: list[str] = []

# The plugin source. Note: only `PluginSnippet` and `BlockedByPlugin` are
# auto-injected into the namespace (see management.plugins.compile_plugin_class).
STREAM_TAPPER = """\
from gateway.tests.test_stream_plugin_demo import CALLED

class StreamTapper(PluginSnippet):
    def before_request(self, request, token, body):
        CALLED.append("before_request")

    def on_stream_chunk(self, chunk):
        # chunk is a litellm ModelResponseStream; grab the text delta.
        pieces = []
        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            pieces.append(delta.get("content") or "")
        CALLED.append("on_stream_chunk")
        print("  [plugin] on_stream_chunk ->", "".join(pieces))
        return chunk

    def after_response(self, request, token, response):
        CALLED.append("after_response")
        print("  [plugin] after_response ->", type(response).__name__)

    def on_error(self, request, exc):
        CALLED.append("on_error")
        print("  [plugin] on_error ->", exc)
"""


@override_settings(TIKA_SERVER_URL=None)
class StreamPluginDemo(ChatCompletionsBase):
    MESSAGES: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hello"}]

    async def test_streaming_plugin_demo(self):
        _plugin_class.cache_clear()
        CALLED.clear()
        await Snippet.objects.acreate(
            name="stream-tapper", type=SnippetType.PLUGIN, active=True, order=0, code=STREAM_TAPPER
        )

        print("\n=== sending streaming chat completion ===")
        response = await self._send_chat_completion_streaming(self.MESSAGES)
        print("=== response status:", response.status_code)

        streamed_lines = await _read_streaming_response_lines(response)
        print(f"=== received {len(streamed_lines)} SSE data lines ===")

        print("\n=== plugin hooks that ran (in order) ===")
        for name in CALLED:
            print("  *", name)

        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertIn("on_stream_chunk", CALLED, "on_stream_chunk never fired")
        self.assertIn("before_request", CALLED)
        self.assertIn("after_response", CALLED)

        _plugin_class.cache_clear()
