from django.core.exceptions import ValidationError
from django.test import TestCase

from management.models import Snippet, SnippetType
from management.plugins import (
    BlockedByPlugin,
    Plugin,
    _error_hook,
    _plugin_class,
    after_hook,
    before_hook,
    compile_plugin_class,
    resolve_active_plugins,
)

GOOD_PLUGIN = """\
class MyPlugin(Plugin):
    def before_request(self, request, token, body):
        body["flagged"] = True
        return body

    def after_response(self, request, token, response):
        self.seen = response
"""

BAD_SYNTAX = "class MyPlugin(Plugin):\n   def before_request(self"


class PluginBaseTestCase(TestCase):
    def test_default_hooks_are_no_ops(self):
        p = Plugin()
        self.assertIsNone(p.before_request("r", "t", "b"))
        self.assertIsNone(p.after_response("r", "t", "resp"))
        self.assertIsNone(p.on_error("r", ValueError("boom")))

    def test_blocked_by_plugin_defaults(self):
        e = BlockedByPlugin("nope")
        self.assertEqual(e.reason, "nope")
        self.assertEqual(e.status, 403)
        self.assertEqual(str(e), "nope")

    def test_blocked_by_plugin_custom_status(self):
        e = BlockedByPlugin("nope", status=429)
        self.assertEqual(e.status, 429)


class CompilePluginTestCase(TestCase):
    def test_compiles_valid_plugin(self):
        cls = compile_plugin_class(GOOD_PLUGIN)
        self.assertTrue(issubclass(cls, Plugin))
        self.assertIsNot(cls, Plugin)

    def test_base_class_auto_injected(self):
        cls = compile_plugin_class("class P(Plugin):\n    pass\n")
        self.assertTrue(issubclass(cls, Plugin))

    def test_rejects_syntax_error(self):
        with self.assertRaises(ValidationError):
            compile_plugin_class(BAD_SYNTAX)

    def test_rejects_exec_error(self):
        with self.assertRaises(ValidationError):
            compile_plugin_class("raise RuntimeError('boom')")

    def test_rejects_no_subclass(self):
        with self.assertRaises(ValidationError):
            compile_plugin_class("x = 1")

    def test_rejects_non_subclass(self):
        with self.assertRaises(ValidationError):
            compile_plugin_class("class Anything:\n    pass\n")

    def test_rejects_multiple_subclasses(self):
        with self.assertRaises(ValidationError):
            compile_plugin_class("class A(Plugin):\n    pass\n\nclass B(Plugin):\n    pass\n")


class ResolvePluginsTestCase(TestCase):
    def setUp(self):
        _plugin_class.cache_clear()

    def _plugin(self, name: str, cap: str, active: bool = True, order: int = 0) -> Snippet:
        return Snippet.objects.create(
            name=name,
            type=SnippetType.PLUGIN,
            active=active,
            order=order,
            code=f"class P(Plugin):\n    def after_response(self, r, t, resp):\n        {cap}\n",
        )

    def test_no_active_plugins_by_default(self):
        self.assertEqual(resolve_active_plugins(), [])

    def test_only_active_plugins_resolved(self):
        self._plugin("on", "pass", active=True)
        self._plugin("off", "pass", active=False)
        plugins = resolve_active_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(type(plugins[0]).__name__, "P")

    def test_config_snippets_are_not_plugins(self):
        Snippet.objects.create(
            name="cfg",
            type=SnippetType.CONFIG,
            active=True,
            code="class C(ConfigSnippet):\n    pass\n",
        )
        self.assertEqual(resolve_active_plugins(), [])

    def test_ordered_by_order_pk(self):
        self._plugin("first", "pass", order=2)
        self._plugin("second", "pass", order=0)
        self._plugin("third", "pass", order=1)
        expected = list(
            Snippet.objects.filter(type=SnippetType.PLUGIN, active=True)
            .order_by("order", "id")
            .values_list("pk", flat=True)
        )
        instances = resolve_active_plugins()
        for pk in expected:
            _plugin_class(pk)
        self.assertIsInstance(instances[0], Plugin)
        self.assertEqual(len(instances), len(expected))

    def test_fresh_instance_per_call_and_class_cached(self):
        self._plugin("on", "pass")
        plugins_a = resolve_active_plugins()
        plugins_b = resolve_active_plugins()
        self.assertIsNot(plugins_a[0], plugins_b[0])
        self.assertIs(type(plugins_a[0]), type(plugins_b[0]))


class DispatchPluginsTestCase(TestCase):
    def _make(self, before=None, after=None, on_error=None) -> Plugin:
        class P(Plugin):
            pass

        if before is not None:
            P.before_request = staticmethod(before)
        if after is not None:
            P.after_response = staticmethod(after)
        if on_error is not None:
            P.on_error = staticmethod(on_error)
        return P()

    def test_before_hook_pipeline(self):
        calls = []

        def a(request, token, body):
            calls.append("a")
            body["a"] = 1
            return body

        def b(request, token, body):
            calls.append("b")

        plugins = [self._make(before=a), self._make(before=b)]
        body = {"x": 0}
        result = before_hook(plugins, "req", "tok", body)
        self.assertEqual(result, {"x": 0, "a": 1})
        self.assertEqual(calls, ["a", "b"])

    def test_before_hook_blocked_propagates(self):
        def blocker(request, token, body):
            raise BlockedByPlugin("denied")

        reached = []

        def a(request, token, body):
            reached.append("a")
            return body

        plugins = [
            self._make(before=a),
            self._make(before=blocker),
            self._make(before=lambda r, t, b: None),
        ]
        with self.assertRaises(BlockedByPlugin) as ctx:
            before_hook(plugins, "req", "tok", {})
        self.assertEqual(ctx.exception.reason, "denied")
        self.assertEqual(reached, ["a"])

    def test_before_hook_records_processing_time(self):
        p = self._make(before=lambda r, t, b: b)
        before_hook([p], "req", "tok", {})
        self.assertIsInstance(p.processing_time_ms, int)

    def test_after_hook_runs_all(self):
        calls = []

        def a(request, token, response):
            calls.append(("a", response))

        def b(request, token, response):
            calls.append(("b", response))

        plugins = [self._make(after=a), self._make(after=b)]
        after_hook(plugins, "req", "tok", "resp")
        self.assertEqual(calls, [("a", "resp"), ("b", "resp")])

    def test_error_hook_receives_exception(self):
        seen = []
        e = ValueError("boom")

        def handler(request, exc):
            seen.append(exc)

        _error_hook([self._make(on_error=handler)], "req", e)
        self.assertEqual(seen, [e])
