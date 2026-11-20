import json
import tempfile
from http import HTTPStatus
from pathlib import Path

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Permission
from django.test import TestCase, override_settings
from django.urls import reverse

from gateway.config import get_mcp_config, model_names
from gateway.tests.utils.base import ROUTER_CONFIG_PATH
from management.admin import OrgAdminForm, TeamAdminForm, UserProfileAdminForm
from management.models import Org, Team
from management.tests.helpers import make_token, make_user, router_config

User = get_user_model()
MCP_SERVERS = ("aqueduct", "docs")


def posted(form) -> dict:
    data = {name: form[name].value() for name in form.fields}
    data |= {
        f"{name}-shown": [value for value, _ in form.fields[name].choices]
        for name in form.fields
        if name.startswith("available_")
    }
    return {name: value for name, value in data.items() if value not in (None, False)}


class AvailabilityTestCase(TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "mcp.json"
        path.write_text(json.dumps({"mcpServers": {name: {"url": "x"} for name in MCP_SERVERS}}))
        settings = override_settings(
            MCP_CONFIG_FILE_PATH=str(path), LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG_PATH
        )
        settings.enable()
        self.addCleanup(settings.disable)
        get_mcp_config.cache_clear()
        self.addCleanup(get_mcp_config.cache_clear)
        self.team = Team.objects.create(name="Whale", org=Org.objects.create(name="E060"))


class TeamAvailabilityFormTest(AvailabilityTestCase):
    def test_every_box_starts_checked(self):
        form = TeamAdminForm(instance=self.team)
        self.assertEqual(form["available_models"].value(), list(model_names()))
        self.assertEqual(form["available_mcp_servers"].value(), list(MCP_SERVERS))

    def test_checked_boxes_are_what_stays_available(self):
        self.team.excluded_models = ["whisper-1"]
        self.team.excluded_mcp_servers = ["docs"]
        self.team.save()
        form = TeamAdminForm(instance=self.team)
        self.assertNotIn("whisper-1", form["available_models"].value())
        self.assertEqual(form["available_mcp_servers"].value(), ["aqueduct"])

    def test_saving_unchanged_keeps_what_is_stored(self):
        self.team.excluded_models = ["retired-model", "whisper-1"]
        self.team.excluded_mcp_servers = ["retired-server", "aqueduct"]
        self.team.save()
        form = TeamAdminForm(posted(TeamAdminForm(instance=self.team)), instance=self.team)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, ["retired-model", "whisper-1"])
        self.assertEqual(self.team.excluded_mcp_servers, ["retired-server", "aqueduct"])

    def test_clearing_a_box_makes_it_unavailable_and_unknown_names_stay(self):
        self.team.excluded_models = ["retired-model"]
        self.team.save()
        data = posted(TeamAdminForm(instance=self.team))
        data["available_models"].remove("dall-e-2")
        data["available_mcp_servers"].remove("docs")
        form = TeamAdminForm(data, instance=self.team)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, ["retired-model", "dall-e-2"])
        self.assertEqual(self.team.excluded_mcp_servers, ["docs"])

    def test_checking_a_box_makes_it_available_again(self):
        self.team.excluded_models = ["retired-model", "whisper-1"]
        self.team.save()
        data = posted(TeamAdminForm(instance=self.team))
        data["available_models"].append("whisper-1")
        form = TeamAdminForm(data, instance=self.team)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, ["retired-model"])

    def test_a_list_the_form_did_not_render_stays_as_stored(self):
        self.team.excluded_models = ["whisper-1"]
        self.team.save()
        data = posted(TeamAdminForm(instance=self.team))
        del data["available_models"], data["available_models-shown"]
        form = TeamAdminForm(data, instance=self.team)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, ["whisper-1"])

    def test_a_cleared_list_makes_every_model_unavailable(self):
        data = posted(TeamAdminForm(instance=self.team))
        del data["available_models"]
        form = TeamAdminForm(data, instance=self.team)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, list(model_names()))


class RepeatedModelNameTest(AvailabilityTestCase):
    def entries(self, names: str) -> list[dict]:
        return [{"model_name": n, "litellm_params": {"model": f"openai/{n}"}} for n in names]

    def test_a_model_added_while_the_form_is_open_stays_available(self):
        with router_config(self.entries("aab")):
            data = posted(TeamAdminForm(instance=self.team))
        data["available_models"] = ["b"]
        with router_config(self.entries("abc")):
            form = TeamAdminForm(data, instance=self.team)
            self.assertTrue(form.is_valid(), form.errors)
            form.save()
        self.team.refresh_from_db()
        self.assertEqual(self.team.excluded_models, ["a"])

    def test_a_repeated_name_is_one_box(self):
        with router_config(self.entries("aab")):
            form = TeamAdminForm(instance=self.team)
            self.assertEqual(form["available_models"].value(), ["a", "b"])


class ProfileAvailabilityFormTest(AvailabilityTestCase):
    def test_a_cleared_box_is_what_the_gateway_refuses(self):
        user = make_user("member", self.team.org)
        data = posted(UserProfileAdminForm(instance=user.profile))
        data["available_models"].remove("whisper-1")
        form = UserProfileAdminForm(data, instance=user.profile)
        self.assertTrue(form.is_valid(), form.errors)
        form.save()
        key = make_token(User.objects.get(pk=user.pk), "laptop")
        self.assertTrue(key.model_excluded("whisper-1"))
        self.assertFalse(key.model_excluded("dall-e-2"))


class OrgAvailabilityAdminPageTest(AvailabilityTestCase):
    def setUp(self):
        super().setUp()
        self.org = self.team.org
        self.org.excluded_models = ["retired-model"]
        self.org.save()
        admin = User.objects.create_superuser("root", "root@example.com", "pw")
        self.client.force_login(admin)
        self.url = reverse("admin:management_org_change", args=[self.org.pk])

    def test_posting_the_admin_form_writes_the_exclusion_lists(self):
        form = self.client.get(self.url).context["adminform"].form
        self.assertIsInstance(form, OrgAdminForm)
        data = posted(form)
        data["available_models"].remove("dall-e-2")
        data["available_mcp_servers"].remove("docs")
        response = self.client.post(self.url, data)
        self.assertEqual(response.status_code, HTTPStatus.FOUND)
        self.org.refresh_from_db()
        self.assertEqual(self.org.excluded_models, ["retired-model", "dall-e-2"])
        self.assertEqual(self.org.excluded_mcp_servers, ["docs"])
        self.assertTrue(self.org.merge_exclusion_lists)

    def test_org_team_and_user_pages_offer_the_checklists_in_place_of_the_raw_lists(self):
        user = make_user("member", self.org)
        pages = (
            self.url,
            reverse("admin:management_team_change", args=[self.team.pk]),
            reverse("admin:auth_user_change", args=[user.pk]),
        )
        for page in pages:
            with self.subTest(page=page):
                response = self.client.get(page)
                self.assertContains(response, 'available_models"')
                self.assertContains(response, 'available_mcp_servers"')
                self.assertNotContains(response, 'excluded_models"')
                self.assertContains(response, 'concurrent_requests_limit"')


@override_settings(AUTHENTICATION_BACKENDS=["django.contrib.auth.backends.ModelBackend"])
class ViewOnlyAdminTest(AvailabilityTestCase):
    def test_staff_who_may_only_view_see_what_is_available(self):
        org = self.team.org
        org.excluded_models = ["whisper-1"]
        org.excluded_mcp_servers = ["docs"]
        org.save()
        viewer = User.objects.create_user("viewer", "viewer@example.com", "pw", is_staff=True)
        viewer.user_permissions.set(Permission.objects.filter(codename="view_org"))
        self.client.force_login(viewer)

        response = self.client.get(reverse("admin:management_org_change", args=[org.pk]))

        shown = [n for n in model_names() if n != "whisper-1"]
        self.assertContains(response, ", ".join(shown))
        self.assertNotContains(response, "whisper-1")
        self.assertContains(response, ">aqueduct<")
