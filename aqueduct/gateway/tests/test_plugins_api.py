from typing import ClassVar

from django.test import override_settings

from gateway.tests.utils import _read_streaming_response_lines
from gateway.tests.utils.base import ChatCompletionsBase
from management.models import Snippet, SnippetType
from management.plugins import _plugin_class
from mock_api.mock_configs import MockConfig

AFTER_CALLS: list[bool] = []
EVENTS: list[str] = []
STREAM_CALLS: list[str] = []
BODY_EXTRA_KEY = "custom_plugin_field"


BODY_MUTATING_PLUGIN = """\
from gateway.tests.test_plugins_api import BODY_EXTRA_KEY
class Mutator(PluginSnippet):
    def before_request(self, request, token, body):
        body[BODY_EXTRA_KEY] = "plugin-added"
        return body
"""


BLOCKING_PLUGIN = """\
class Guard(PluginSnippet):
    def before_request(self, request, token, body):
        raise BlockedByPlugin("forbidden by guard", status=403)
"""


AFTER_PLUGIN = """\
from gateway.tests.test_plugins_api import AFTER_CALLS
class Audit(PluginSnippet):
    def after_response(self, response):
        AFTER_CALLS.append(True)
"""


PLUGIN_RECORD_A = """\
from gateway.tests.test_plugins_api import EVENTS
class A(PluginSnippet):
    def before_request(self, request, token, body):
        EVENTS.append("A")
        return body
"""


PLUGIN_RECORD_B = """\
from gateway.tests.test_plugins_api import EVENTS
class B(PluginSnippet):
    def before_request(self, request, token, body):
        EVENTS.append("B")
        return body
"""


PLUGIN_SET_STEP = """\
from gateway.tests.test_plugins_api import EVENTS
class SetStep(PluginSnippet):
    def before_request(self, request, token, body):
        body["step"] = "from-pipe"
        return body
"""


PLUGIN_READ_STEP = """\
from gateway.tests.test_plugins_api import EVENTS
class ReadStep(PluginSnippet):
    def before_request(self, request, token, body):
        EVENTS.append(body.get("step"))
        return body
"""


PLUGIN_BOOM = """\
class Boom(PluginSnippet):
    def before_request(self, request, token, body):
        raise RuntimeError("plugin exploded")
"""


STREAM_PLUGIN = """\
from gateway.tests.test_plugins_api import STREAM_CALLS
class Wrap(PluginSnippet):
    def after_streaming_response(self, response):
        STREAM_CALLS.append("after_streaming_response")
        return response
"""


AFTER_REPLACING_PLUGIN = """\
from gateway.response_type import RawJsonResponse
from gateway.tests.test_plugins_api import AFTER_CALLS
class Replace(PluginSnippet):
    def after_response(self, response):
        return RawJsonResponse({"plugin": "replaced", "kept": True})
"""


ENDPOINT_SCOPED_PLUGIN = """\
from gateway.tests.test_plugins_api import EVENTS
class Scoped(PluginSnippet):
    endpoints = ["chat/completions"]
    def before_request(self, request, token, body):
        EVENTS.append("chat/completions")
        return body
"""


ENDPOINT_MISMATCH_PLUGIN = """\
from gateway.tests.test_plugins_api import EVENTS
class Mismatch(PluginSnippet):
    endpoints = ["embeddings"]
    def before_request(self, request, token, body):
        EVENTS.append("embeddings")
        return body
"""


def seed_plugin(name: str, code: str, order: int = 0) -> Snippet:
    return Snippet.objects.create(
        name=name, type=SnippetType.PLUGIN, active=True, order=order, code=code
    )


@override_settings(TIKA_SERVER_URL=None)
class PluginGatewayIntegrationTest(ChatCompletionsBase):
    MESSAGES: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hello"}]

    def setUp(self):
        super().setUp()
        _plugin_class.cache_clear()
        AFTER_CALLS.clear()
        EVENTS.clear()
        STREAM_CALLS.clear()

    def tearDown(self):
        _plugin_class.cache_clear()
        super().tearDown()

    def test_no_plugins_keeps_response_unchanged(self):
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(AFTER_CALLS)

    def test_before_hook_transforms_body(self):
        seed_plugin("mutate", BODY_MUTATING_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)

    def test_plugin_mutation_reaches_upstream(self):
        """A before_request mutation must be visible in the forwarded LLM body."""
        seed_plugin("mutate", BODY_MUTATING_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        forwarded = self.mock_server.get_last_request("chat/completions")
        self.assertIsNotNone(forwarded, "The request should reach the upstream mock.")
        self.assertEqual(forwarded[BODY_EXTRA_KEY], "plugin-added")

    def test_before_hook_can_block(self):
        seed_plugin("guard", BLOCKING_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(resp.json()["error"]["message"], "forbidden by guard")

    def test_after_hook_observes_response(self):
        seed_plugin("audit", AFTER_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(AFTER_CALLS, [True])

    def test_router_error_still_converted_by_catch(self):
        seed_plugin("audit", AFTER_PLUGIN)
        bad = MockConfig(
            status_code=400,
            response_data={"error": {"message": "upstream boom", "type": "invalid_request_error"}},
        )
        with self.mock_server.patch_external_api("chat/completions", bad):
            resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("upstream boom", resp.json()["error"]["message"])

    def test_plugins_execute_in_order(self):
        seed_plugin("first", PLUGIN_RECORD_A, order=0)
        seed_plugin("second", PLUGIN_RECORD_B, order=1)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, ["A", "B"])

    def test_plugin_execution_order_follows_order_field(self):
        seed_plugin("later", PLUGIN_RECORD_A, order=1)
        seed_plugin("earlier", PLUGIN_RECORD_B, order=0)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, ["B", "A"])

    def test_blocking_plugin_short_circuits_later_plugins(self):
        seed_plugin("guard", BLOCKING_PLUGIN, order=0)
        seed_plugin("audit", PLUGIN_RECORD_B, order=1)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 403)
        self.assertEqual(EVENTS, [], "later plugin should not run once an earlier one blocks")

    def test_before_pipeline_transforms_for_next_plugin(self):
        seed_plugin("set", PLUGIN_SET_STEP, order=0)
        seed_plugin("read", PLUGIN_READ_STEP, order=1)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, ["from-pipe"])

    def test_inactive_plugin_does_not_run(self):
        Snippet.objects.create(
            name="off", type=SnippetType.PLUGIN, active=False, code=PLUGIN_RECORD_A
        )
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, [])

    def test_generic_exception_in_before_becomes_error(self):
        seed_plugin("boom", PLUGIN_BOOM)
        self.client.raise_request_exception = False
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 500)

    def test_after_hook_replaces_response(self):
        seed_plugin("replace", AFTER_REPLACING_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["plugin"], "replaced")

    def test_endpoint_scoped_plugin_runs_on_listed_endpoint(self):
        seed_plugin("scoped", ENDPOINT_SCOPED_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, ["chat/completions"])

    def test_endpoint_scoped_plugin_skipped_on_other_endpoint(self):
        seed_plugin("mismatch", ENDPOINT_MISMATCH_PLUGIN)
        resp = self._send_chat_completion(self.MESSAGES)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(EVENTS, [], "plugin scoped to another endpoint must not run")

    async def test_streaming_plugin_wraps_response(self):
        await Snippet.objects.acreate(
            name="wrap", type=SnippetType.PLUGIN, active=True, order=0, code=STREAM_PLUGIN
        )
        response = await self._send_chat_completion_streaming(self.MESSAGES)
        self.assertEqual(response.status_code, 200)
        streamed_lines = await _read_streaming_response_lines(response)
        self.assertGreater(len(streamed_lines), 0, "Should receive at least one data chunk.")
        self.assertIn("after_streaming_response", STREAM_CALLS)
