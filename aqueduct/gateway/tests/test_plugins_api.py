from typing import ClassVar

from django.test import override_settings

from gateway.tests.test_endpoints import ChatCompletionsBase
from management.models import Snippet, SnippetType
from management.plugins import _plugin_class
from mock_api.mock_configs import MockConfig

AFTER_CALLS: list[bool] = []
EVENTS: list[str] = []
BODY_EXTRA_KEY = "custom_plugin_field"


BODY_MUTATING_PLUGIN = """\
from gateway.tests.test_plugins_api import BODY_EXTRA_KEY
class Mutator(Plugin):
    def before_request(self, request, token, body):
        body[BODY_EXTRA_KEY] = "plugin-added"
        return body
"""


BLOCKING_PLUGIN = """\
class Guard(Plugin):
    def before_request(self, request, token, body):
        raise BlockedByPlugin("forbidden by guard", status=403)
"""


AFTER_PLUGIN = """\
from gateway.tests.test_plugins_api import AFTER_CALLS
class Audit(Plugin):
    def after_response(self, request, token, response):
        AFTER_CALLS.append(True)
"""


PLUGIN_RECORD_A = """\
from gateway.tests.test_plugins_api import EVENTS
class A(Plugin):
    def before_request(self, request, token, body):
        EVENTS.append("A")
        return body
"""


PLUGIN_RECORD_B = """\
from gateway.tests.test_plugins_api import EVENTS
class B(Plugin):
    def before_request(self, request, token, body):
        EVENTS.append("B")
        return body
"""


PLUGIN_SET_STEP = """\
from gateway.tests.test_plugins_api import EVENTS
class SetStep(Plugin):
    def before_request(self, request, token, body):
        body["step"] = "from-pipe"
        return body
"""


PLUGIN_READ_STEP = """\
from gateway.tests.test_plugins_api import EVENTS
class ReadStep(Plugin):
    def before_request(self, request, token, body):
        EVENTS.append(body.get("step"))
        return body
"""


PLUGIN_BOOM = """\
class Boom(Plugin):
    def before_request(self, request, token, body):
        raise RuntimeError("plugin exploded")
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
