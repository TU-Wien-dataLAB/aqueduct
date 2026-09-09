"""Tests for the snippet resolver module (compile/validate/cache/fallback)."""

from django.core.exceptions import ValidationError
from django.test import TestCase

from management.models import Snippet, SnippetType
from management.snippets import ConfigSnippet, compile_snippet, get_config_snippet
from management.tests.helpers import SNIPPET_ORG_CUSTOM, seed_active_config


class CompileSnippetTestCase(TestCase):
    """Test compile_snippet: validation and resulting behavior."""

    def test_compiles_valid_snippet(self):
        s = compile_snippet(SNIPPET_ORG_CUSTOM)
        self.assertEqual(s.org_name({"custom_org": "acme"}), "acme")
        # Unoverridden methods fall back to the base class.
        self.assertEqual(s.team_names({}), [])
        self.assertEqual(s.display_team_names(["a"]), [("a", "a")])
        self.assertEqual(s.user_group({}), "user")

    def test_base_class_auto_injected(self):
        s = compile_snippet("class C(ConfigSnippet):\n    pass\n")
        self.assertIsInstance(s, ConfigSnippet)

    def test_stdlib_import_permitted(self):
        code = """
import json

class C(ConfigSnippet):
    def org_name(self, claims):
        return json.loads(claims.get("org_payload", "{}")).get("org")
"""
        s = compile_snippet(code)
        self.assertEqual(s.org_name({"org_payload": '{"org": "acme"}'}), "acme")

    def test_rejects_syntax_error(self):
        with self.assertRaises(ValidationError):
            compile_snippet("class C(ConfigSnippet):\n   def broken(self")

    def test_rejects_exec_error(self):
        with self.assertRaises(ValidationError):
            compile_snippet("raise RuntimeError('boom')")

    def test_rejects_no_subclass(self):
        with self.assertRaises(ValidationError):
            compile_snippet("x = 1")

    def test_rejects_non_subclass(self):
        with self.assertRaises(ValidationError):
            compile_snippet("class Plain:\n    pass\n")

    def test_rejects_multiple_subclasses(self):
        code = """
class A(ConfigSnippet):
    pass

class B(ConfigSnippet):
    pass
"""
        with self.assertRaises(ValidationError):
            compile_snippet(code)

    def test_rejects_wrong_signature(self):
        code = """
class C(ConfigSnippet):
    def org_name(self):
        return "x"
"""
        with self.assertRaises(ValidationError) as ctx:
            compile_snippet(code)
        self.assertIn("org_name", str(ctx.exception))

    def test_rejects_staticmethod(self):
        code = """
class C(ConfigSnippet):
    @staticmethod
    def org_name(claims):
        return "x"
"""
        with self.assertRaises(ValidationError):
            compile_snippet(code)


class ConfigSnippetDefaultsTestCase(TestCase):
    def test_defaults(self):
        s = ConfigSnippet()
        self.assertEqual(s.org_name({}), "default")
        self.assertEqual(s.team_names({}), [])
        self.assertEqual(s.display_team_names(["a"]), [("a", "a")])
        self.assertEqual(s.user_group({}), "user")


class ResolverTestCase(TestCase):
    """Test get_config_snippet fallback + resolution."""

    def test_default_class_when_no_row(self):
        self.assertEqual(get_config_snippet().org_name({}), "default")

    def test_active_row_overrides_default(self):
        seed_active_config(SNIPPET_ORG_CUSTOM)
        self.assertEqual(get_config_snippet().org_name({"custom_org": "x"}), "x")

    def test_inactive_row_ignored(self):
        Snippet.objects.create(
            name="c", type=SnippetType.CONFIG, active=False, code=SNIPPET_ORG_CUSTOM
        )
        self.assertEqual(get_config_snippet().org_name({}), "default")

    def test_plugin_row_not_used_as_config(self):
        Snippet.objects.create(
            name="p", type=SnippetType.PLUGIN, active=True, code=SNIPPET_ORG_CUSTOM
        )
        self.assertEqual(get_config_snippet().org_name({}), "default")

    def test_each_call_returns_a_fresh_instance(self):
        seed_active_config(SNIPPET_ORG_CUSTOM)
        self.assertIsNot(get_config_snippet(), get_config_snippet())
        self.assertEqual(
            get_config_snippet().org_name({"custom_org": "one"}),
            get_config_snippet().org_name({"custom_org": "one"}),
        )

    def test_cache_refresh_when_code_edited(self):
        row = seed_active_config(SNIPPET_ORG_CUSTOM)
        self.assertEqual(get_config_snippet().org_name({"custom_org": "one"}), "one")

        row.code = (
            'class C(ConfigSnippet):\n    def org_name(self, claims):\n        return "fixed"\n'
        )
        row.save()

        self.assertEqual(get_config_snippet().org_name({"custom_org": "one"}), "fixed")
