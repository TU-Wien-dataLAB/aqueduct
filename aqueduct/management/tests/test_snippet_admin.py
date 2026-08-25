"""Tests for the Snippet admin surface (validation, superuser gating, console)."""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from management.admin import SnippetAdminForm
from management.models import Snippet, SnippetType
from management.tests.helpers import (
    SNIPPET_ORG_CUSTOM,
    SNIPPET_ORG_FROM_FIRST_GROUP,
    seed_active_config,
)

User = get_user_model()

SUPERUSER_EMAIL = "you@example.com"

VALID_CODE = """
class C(ConfigSnippet):
    def org_name(self, claims):
        return claims.get("org")
"""


@override_settings(ADMIN_SUPERUSER_EMAILS=[SUPERUSER_EMAIL])
class SnippetAdminFormTestCase(TestCase):
    def test_valid_code_passes(self):
        form = SnippetAdminForm(
            data={"name": "c", "type": "config", "active": True, "code": SNIPPET_ORG_CUSTOM}
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_syntax_error_rejected(self):
        form = SnippetAdminForm(
            data={
                "name": "c",
                "type": "config",
                "active": True,
                "code": "class C(ConfigSnippet):\n  def bad(self",
            }
        )
        self.assertFalse(form.is_valid())

    def test_wrong_signature_rejected(self):
        form = SnippetAdminForm(
            data={
                "name": "c",
                "type": "config",
                "active": True,
                "code": "class C(ConfigSnippet):\n    def org_name(self):\n        return 'x'\n",
            }
        )
        self.assertFalse(form.is_valid())
        self.assertIn("org_name", str(form.errors["code"]))

    def test_second_active_config_demotes_previous(self):
        seed_active_config(SNIPPET_ORG_CUSTOM)

        form = SnippetAdminForm(
            data={"name": "c2", "type": "config", "active": True, "code": VALID_CODE}
        )
        self.assertTrue(form.is_valid(), form.errors)
        form.save()

        self.assertTrue(
            Snippet.objects.filter(name="c2", type=SnippetType.CONFIG, active=True).exists()
        )
        self.assertFalse(
            Snippet.objects.filter(type=SnippetType.CONFIG, active=True).exclude(name="c2").exists()
        )

    def test_editing_active_config_kept_active_is_ok(self):
        existing = seed_active_config(SNIPPET_ORG_CUSTOM)

        form = SnippetAdminForm(
            data={"name": existing.name, "type": "config", "active": True, "code": VALID_CODE},
            instance=existing,
        )
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.assertEqual(Snippet.objects.filter(type=SnippetType.CONFIG, active=True).count(), 1)

    def test_inactive_config_allowed_alongside_active(self):
        seed_active_config(SNIPPET_ORG_CUSTOM)

        form = SnippetAdminForm(
            data={"name": "c2", "type": "config", "active": False, "code": VALID_CODE}
        )
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.assertEqual(Snippet.objects.filter(type=SnippetType.CONFIG, active=True).count(), 1)

    def test_promoting_config_demotes_all_previous_active(self):
        seed_active_config(SNIPPET_ORG_CUSTOM)
        Snippet.objects.create(name="old1", type=SnippetType.CONFIG, active=True, code=VALID_CODE)
        Snippet.objects.create(name="old2", type=SnippetType.CONFIG, active=True, code=VALID_CODE)

        form = SnippetAdminForm(
            data={"name": "c2", "type": "config", "active": True, "code": VALID_CODE}
        )
        self.assertTrue(form.is_valid(), form.errors)
        form.save()

        active = Snippet.objects.filter(type=SnippetType.CONFIG, active=True)
        self.assertEqual(active.count(), 1)
        self.assertEqual(active.get().name, "c2")

    def test_two_active_plugins_same_name_allowed(self):
        Snippet.objects.create(name="plug", type=SnippetType.PLUGIN, active=True, code=VALID_CODE)
        form = SnippetAdminForm(
            data={"name": "plug", "type": "plugin", "active": True, "code": VALID_CODE}
        )
        self.assertTrue(form.is_valid(), form.errors)

    def test_orm_create_active_config_demotes_previous(self) -> None:
        seed_active_config(SNIPPET_ORG_CUSTOM)
        Snippet.objects.create(name="c2", type=SnippetType.CONFIG, active=True, code=VALID_CODE)

        active = Snippet.objects.filter(type=SnippetType.CONFIG, active=True)
        self.assertEqual(active.count(), 1)
        self.assertEqual(active.get().name, "c2")

    def test_updating_config_to_active_demotes_previous(self) -> None:
        seed_active_config(SNIPPET_ORG_CUSTOM)
        existing = Snippet.objects.create(
            name="c2", type=SnippetType.CONFIG, active=False, code=VALID_CODE
        )

        existing.active = True
        existing.save()

        active = Snippet.objects.filter(type=SnippetType.CONFIG, active=True)
        self.assertEqual(active.count(), 1)
        self.assertEqual(active.get().name, "c2")

    def test_activating_config_does_not_touch_active_plugins(self) -> None:
        seed_active_config(SNIPPET_ORG_CUSTOM)
        plugin = Snippet.objects.create(
            name="p1", type=SnippetType.PLUGIN, active=True, code=VALID_CODE
        )

        config = Snippet.objects.create(
            name="new-config", type=SnippetType.CONFIG, active=True, code=VALID_CODE
        )

        self.assertTrue(config.active)
        # The newly-activated config demoted the old one (only 1 active config).
        self.assertEqual(Snippet.objects.filter(type=SnippetType.CONFIG, active=True).count(), 1)
        # Active plugins are left untouched.
        plugin.refresh_from_db()
        self.assertTrue(plugin.active)
        self.assertEqual(Snippet.objects.filter(type=SnippetType.PLUGIN, active=True).count(), 1)


@override_settings(ADMIN_SUPERUSER_EMAILS=[SUPERUSER_EMAIL])
class SnippetAdminAuthorizationTestCase(TestCase):
    def setUp(self):
        self.superuser = User.objects.create_superuser(
            username="admin", email=SUPERUSER_EMAIL, password="pw"
        )
        self.staff = User.objects.create_user(
            username="staff", email="staff@example.com", password="pw", is_staff=True
        )
        self.changelist_url = reverse("admin:management_snippet_changelist")
        self.console_url = reverse("admin:management_snippet_test_console")

    def test_superuser_can_view_changelist(self):
        self.client.force_login(self.superuser)
        resp = self.client.get(self.changelist_url)
        self.assertEqual(resp.status_code, 200)

    def test_non_superuser_staff_forbidden_on_changelist(self):
        self.client.force_login(self.staff)
        resp = self.client.get(self.changelist_url)
        self.assertEqual(resp.status_code, 403)

    def test_non_superuser_forbidden_on_console(self):
        self.client.force_login(self.staff)
        resp = self.client.get(self.console_url)
        self.assertEqual(resp.status_code, 403)

    def test_anonymous_forbidden_on_console(self):
        resp = self.client.get(self.console_url)
        # Anonymous users are redirected to the login page (or 403), never 200.
        self.assertNotEqual(resp.status_code, 200)

    def test_superuser_not_in_allowlist_forbidden_on_changelist(self):
        other = User.objects.create_superuser(
            username="other", email="other@example.com", password="pw"
        )
        self.client.force_login(other)
        resp = self.client.get(self.changelist_url)
        self.assertEqual(resp.status_code, 403)


@override_settings(ADMIN_SUPERUSER_EMAILS=[SUPERUSER_EMAIL])
class SnippetConsoleTestCase(TestCase):
    def setUp(self):
        self.superuser = User.objects.create_superuser(
            username="admin", email=SUPERUSER_EMAIL, password="pw"
        )
        self.client.force_login(self.superuser)
        self.url = reverse("admin:management_snippet_test_console")

    def test_get_shows_console(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Snippet test console")
        self.assertContains(resp, "executes real server code")

    def test_get_prefills_example_payload(self):
        resp = self.client.get(self.url)
        self.assertEqual(resp.status_code, 200)
        # The Test input is pre-filled with a runnable claims example.
        self.assertContains(resp, '"email"')
        self.assertContains(resp, "you@example.com")
        self.assertContains(resp, '"groups"')
        self.assertContains(resp, "E123-Students")
        # Dimmed on-page example blocks + large editor are present.
        self.assertContains(resp, "Example claims to paste here")
        self.assertContains(resp, "snippet-console-example")
        self.assertContains(resp, "snippet-console-code")

    def test_runs_methods_against_payload(self):
        resp = self.client.post(
            self.url,
            {"code": SNIPPET_ORG_FROM_FIRST_GROUP, "payload": json.dumps({"groups": ["acme"]})},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Test complete.")
        self.assertContains(resp, "org_name(claims)")
        self.assertContains(resp, "acme")
        self.assertContains(resp, "user_group(claims)")

    def test_rejects_invalid_code(self):
        resp = self.client.post(
            self.url, {"code": "class C(ConfigSnippet):\n  oops", "payload": "{}"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "failed to compile")

    def test_rejects_invalid_json(self):
        resp = self.client.post(
            self.url, {"code": SNIPPET_ORG_FROM_FIRST_GROUP, "payload": "{not json"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Invalid JSON test input")
