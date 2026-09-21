from typing import ClassVar

from django.test import override_settings

from gateway.tests.test_endpoints import ChatCompletionsBase
from management.models import Snippet, SnippetType
from management.plugins import plugin_class
from mock_api.mock_configs import MockConfig

AFTER_CALLS: list[bool] = []
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


def seed_plugin(name: str, code: str, order: int = 0) -> Snippet:
    return Snippet.objects.create(
        name=name, type=SnippetType.PLUGIN, active=True, order=order, code=code
    )


@override_settings(TIKA_SERVER_URL=None)
class PluginGatewayIntegrationTest(ChatCompletionsBase):
    MESSAGES: ClassVar[list[dict[str, str]]] = [{"role": "user", "content": "hello"}]

    def setUp(self):
        super().setUp()
        plugin_class.cache_clear()
        AFTER_CALLS.clear()

    def tearDown(self):
        plugin_class.cache_clear()
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
