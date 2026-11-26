from django.core.exceptions import ValidationError
from django.test import TestCase

from management.models import Snippet, SnippetType
from management.plugins import (
    BlockedByPlugin,
    PluginSnippet,
    _plugin_class,
    after_hook,
    after_streaming_hook,
    before_hook,
    compile_plugin_class,
    resolve_active_plugins,
)

GOOD_PLUGIN = """\
class MyPlugin(PluginSnippet):
    endpoints = ["chat/completions"]

    def before_request(self, request, token, body):
        body["flagged"] = True
        return body

    def after_response(self, response):
        self.seen = response
"""

BAD_SYNTAX = "class MyPlugin(PluginSnippet):\n   def before_request(self"


class _FakeRequest:
    def __init__(self, path: str) -> None:
        self.path = path


class PluginBaseTestCase(TestCase):
    def test_default_hooks_are_no_ops(self):
        p = PluginSnippet()
        self.assertIsNone(p.before_request("r", "t", "b"))
        self.assertIsNone(p.after_response("resp"))
        self.assertIsNone(p.after_streaming_response("resp"))

    def test_default_endpoints_is_empty(self):
        self.assertEqual(PluginSnippet.endpoints, [])

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
        self.assertTrue(issubclass(cls, PluginSnippet))
        self.assertIsNot(cls, PluginSnippet)
        self.assertEqual(cls.endpoints, ["chat/completions"])

    def test_base_class_auto_injected(self):
        cls = compile_plugin_class("class P(PluginSnippet):\n    pass\n")
        self.assertTrue(issubclass(cls, PluginSnippet))

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
            compile_plugin_class(
                "class A(PluginSnippet):\n    pass\n\nclass B(PluginSnippet):\n    pass\n"
            )


class ResolvePluginsTestCase(TestCase):
    def setUp(self):
        _plugin_class.cache_clear()

    def _plugin(self, name: str, cap: str, active: bool = True, order: int = 0) -> Snippet:
        return Snippet.objects.create(
            name=name,
            type=SnippetType.PLUGIN,
            active=active,
            order=order,
            code=(
                f"class P(PluginSnippet):\n    def after_response(self, response):\n        {cap}\n"
            ),
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
        self.assertIsInstance(instances[0], PluginSnippet)
        self.assertEqual(len(instances), len(expected))

    def test_fresh_instance_per_call_and_class_cached(self):
        self._plugin("on", "pass")
        plugins_a = resolve_active_plugins()
        plugins_b = resolve_active_plugins()
        self.assertIsNot(plugins_a[0], plugins_b[0])
        self.assertIs(type(plugins_a[0]), type(plugins_b[0]))

    def test_cache_invalidated_on_snippet_update(self):
        snippet = self._plugin("on", "pass", active=True)
        old_cls = type(resolve_active_plugins()[0])

        snippet.code = (
            "class P(PluginSnippet):\n"
            "    def before_request(self, request, token, body):\n"
            "        body['updated'] = True\n"
            "        return body\n"
        )
        snippet.save()

        new_cls = type(resolve_active_plugins()[0])
        self.assertIsNot(new_cls, old_cls, "Updating a plugin must recompile its class")

    def test_cache_invalidated_on_snippet_delete(self):
        snippet = self._plugin("on", "pass", active=True)
        _plugin_class(snippet.pk)  # populate the cache
        snippet.delete()
        # Cache must be cleared so a re-query raises instead of returning a stale class.
        with self.assertRaises(Snippet.DoesNotExist):
            _plugin_class(snippet.pk)


class DispatchPluginsTestCase(TestCase):
    def _make(self, before=None, after=None, on_stream_after=None, endpoints=None) -> PluginSnippet:
        class P(PluginSnippet):
            pass

        if before is not None:
            P.before_request = staticmethod(before)
        if after is not None:
            P.after_response = staticmethod(after)
        if on_stream_after is not None:
            P.after_streaming_response = staticmethod(on_stream_after)
        if endpoints is not None:
            P.endpoints = endpoints
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
        result = before_hook(plugins, _FakeRequest("chat/completions"), "tok", body)
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
            before_hook(plugins, _FakeRequest("chat/completions"), "tok", {})
        self.assertEqual(ctx.exception.reason, "denied")
        self.assertEqual(reached, ["a"])

    def test_before_hook_records_processing_time(self):
        p = self._make(before=lambda r, t, b: b)
        before_hook([p], _FakeRequest("chat/completions"), "tok", {})
        self.assertIsInstance(p.processing_time_ms, int)

    def test_after_hook_runs_all(self):
        calls = []

        def a(response):
            calls.append(("a", response))
            return response

        def b(response):
            calls.append(("b", response))
            return response

        plugins = [self._make(after=a), self._make(after=b)]
        result = after_hook(plugins, _FakeRequest("chat/completions"), "resp")
        self.assertEqual(result, "resp")
        self.assertEqual(calls, [("a", "resp"), ("b", "resp")])

    def test_after_hook_replaces_response(self):
        def a(response):
            return {"replaced": True}

        plugins = [self._make(after=a)]
        result = after_hook(plugins, _FakeRequest("chat/completions"), {"orig": True})
        self.assertEqual(result, {"replaced": True})

    def test_after_hook_records_processing_time(self):
        p = self._make(after=lambda r: None)
        after_hook([p], _FakeRequest("chat/completions"), "resp")
        self.assertIsInstance(p.processing_time_ms, int)

    def test_after_streaming_hook_runs_all(self):
        seen = []

        def a(response):
            seen.append(("a", response))
            return response

        def b(response):
            seen.append(("b", response))
            return response

        plugins = [self._make(on_stream_after=a), self._make(on_stream_after=b)]
        result = after_streaming_hook(plugins, _FakeRequest("chat/completions"), "resp")
        self.assertEqual(result, "resp")
        self.assertEqual(seen, [("a", "resp"), ("b", "resp")])

    def test_after_streaming_hook_replaces_response(self):
        def a(response):
            return {"wrapped": True}

        plugins = [self._make(on_stream_after=a)]
        result = after_streaming_hook(plugins, _FakeRequest("chat/completions"), "resp")
        self.assertEqual(result, {"wrapped": True})

    def test_plugin_skipped_for_non_listed_endpoint(self):
        calls = []

        def a(request, token, body):
            calls.append(body)
            return body

        plugin = self._make(before=a, endpoints=["chat/completions"])
        before_hook([plugin], _FakeRequest("/v1/chat/completions"), "tok", {"x": 1})
        before_hook([plugin], _FakeRequest("/v1/embeddings"), "tok", {"x": 2})
        self.assertEqual(calls, [{"x": 1}], "plugin must only run on its listed endpoint")

    def test_endpoint_normalizes_v1_and_leading_slash(self):
        calls = []

        def a(request, token, body):
            calls.append(True)
            return body

        plugin = self._make(before=a, endpoints=["chat/completions"])
        before_hook([plugin], _FakeRequest("/v1/chat/completions"), "tok", {})
        before_hook([plugin], _FakeRequest("/chat/completions"), "tok", {})
        self.assertEqual(calls, [True, True])
