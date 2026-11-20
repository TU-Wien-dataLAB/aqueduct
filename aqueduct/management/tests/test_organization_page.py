from http import HTTPStatus

from django.forms.models import model_to_dict
from django.test import override_settings
from django.urls import reverse

from gateway.config import model_names
from management.availability import configured_mcp_servers
from management.models import Org, Token, UserGroup
from management.ownership import RELATED
from management.tests.helpers import RoleWorld, configured_limits, make_token, mcp_config


def enforced(key: Token) -> Token:
    return Token.objects.select_related(*RELATED).get(pk=key.pk)


LIMITS = {
    "requests_per_minute": "600",
    "input_tokens_per_minute": "",
    "output_tokens_per_minute": "5000",
    "concurrent_requests_limit": "12",
    "hourly_limit_multiplier": "30",
    "daily_limit_multiplier": "",
}


def available(*unavailable: str, mcp: tuple[str, ...] | None = None) -> dict:
    data: dict = {
        "available_models": [m for m in list(model_names()) if m not in unavailable],
        "available_models-shown": list(model_names()),
    }
    if mcp is not None:
        data["available_mcp_servers"] = [s for s in configured_mcp_servers() if s not in mcp]
        data["available_mcp_servers-shown"] = configured_mcp_servers()
    return data


def edits() -> tuple[tuple[str, dict], ...]:
    return (
        ("organization_limits", LIMITS),
        ("organization_availability", available("gpt-4.1-nano")),
    )


class OrganizationTestBase(RoleWorld):
    def setUp(self):
        super().setUp()
        self.far_admin = self.user("far-admin", UserGroup.ADMIN, self.other_org)

    def stored(self, org: Org) -> dict:
        return model_to_dict(Org.objects.get(pk=org.pk))


TIMES = "\N{MULTIPLICATION SIGN}"


class OrganizationPageTest(OrganizationTestBase):
    url = reverse("organization")

    def test_users_and_team_admins_get_404(self):
        for user in (self.member, self.team_admin, self.solo):
            with self.subTest(user=user.username):
                self.assertEqual(self.as_user(user).get(self.url).status_code, 404)

    def test_org_admin_sees_their_org_with_edit_links(self):
        self.org.requests_per_minute = 600
        self.org.save()
        response = self.as_user(self.org_admin).get(self.url)
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(
            response,
            '<h1 class="break-words text-3xl font-semibold">Limits and access</h1>',
            html=True,
        )
        self.assertContains(response, "<span>lab-org</span>", html=True)
        self.assertNotContains(response, "people")
        self.assertContains(response, "2 teams")
        self.assertContains(response, "600")
        self.assertContains(response, reverse("organization_limits", args=[self.org.id]))
        self.assertContains(response, reverse("organization_availability", args=[self.org.id]))
        self.assertNotContains(response, reverse("org_switch"))

    def test_limits_read_by_window_and_the_key_count_stays_a_server_setting(self):
        self.org.requests_per_minute = 10
        self.org.hourly_limit_multiplier = 30
        self.org.save()
        response = self.as_user(self.org_admin).get(self.url)
        limits = configured_limits(response)
        self.assertEqual(limits["Per minute"], ["10", "No limit", "No limit"])
        self.assertEqual(limits["Per hour"], ["300", "No limit", "No limit"])
        self.assertEqual(limits["Per day"], ["14,400", "No limit", "No limit"])
        self.assertContains(response, '<span class="muted">(server setting)</span>')
        self.assertNotContains(response, "full rate")

    def test_the_page_counts_the_teams_and_people_with_limits_of_their_own(self):
        response = self.as_user(self.org_admin).get(self.url)
        self.assertContains(response, "Each key gets these limits unless it has its own.")
        self.assertNotContains(response, "of their own.")
        self.team.concurrent_requests_limit = 2
        self.team.save()
        self.solo.profile.requests_per_minute = 5
        self.solo.profile.save()
        self.member.profile.input_tokens_per_minute = 5
        self.member.profile.save()
        response = self.as_user(self.org_admin).get(self.url)
        self.assertContains(response, "1 team has limits of its own.")
        self.assertContains(response, "2 people have limits of their own.")

    def test_the_form_labels_the_multipliers_per_hour_and_per_day_with_a_worked_sentence(self):
        self.org.requests_per_minute = 10
        self.org.input_tokens_per_minute = 2000
        self.org.hourly_limit_multiplier = 3
        self.org.save()
        form = self.as_user(self.org_admin).get(reverse("organization_limits", args=[self.org.id]))
        self.assertContains(form, "Per hour, as a multiple of per minute</label>")
        self.assertContains(form, "Per day, as a multiple of per minute</label>")
        self.assertContains(form, "Requests per minute</label>")
        self.assertContains(
            form, f"Per hour: Requests 30, Input tokens 6,000 (3 {TIMES} the per-minute limits)."
        )
        figures = "Requests 14,400, Input tokens 2,880,000"
        day = f"Per day: {figures} (1,440 {TIMES} the per-minute limits)."
        self.assertContains(form, day)

    def test_the_sentence_says_so_without_a_per_minute_limit(self):
        form = self.as_user(self.org_admin).get(reverse("organization_limits", args=[self.org.id]))
        self.assertContains(form, "Per hour: no per-minute limit is set, so there is no limit.")

    def test_a_form_with_an_error_keeps_the_numbers_in_the_sentence(self):
        response = self.as_user(self.org_admin).post(
            reverse("organization_limits", args=[self.org.id]),
            {
                "requests_per_minute": "10",
                "hourly_limit_multiplier": "5",
                "daily_limit_multiplier": "9000",
            },
        )
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertContains(response, f"Per hour: Requests 50 (5 {TIMES} the per-minute limits).")
        self.assertContains(
            response, f"Per day: Requests 90,000 (9,000 {TIMES} the per-minute limits)."
        )

    def test_concurrent_requests_show_the_server_default_until_the_org_sets_its_own(self):
        with self.settings(AQUEDUCT_DEFAULT_TOKEN_CONCURRENT_REQUESTS_LIMIT=3):
            response = self.as_user(self.org_admin).get(self.url)
        self.assertContains(response, "No limit on requests or tokens.")
        self.assertEqual(configured_limits(response)["Concurrent requests"], ["3"])
        self.org.concurrent_requests_limit = 12
        self.org.save()
        response = self.as_user(self.org_admin).get(self.url)
        self.assertEqual(configured_limits(response)["Concurrent requests"], ["12"])

    def test_the_concurrent_limit_row_is_left_out_when_the_gateway_does_not_enforce_it(self):
        with self.settings(AQUEDUCT_TOKEN_CONCURRENT_REQUESTS_LIMIT_ENABLED=False):
            response = self.as_user(self.org_admin).get(self.url)
        self.assertNotContains(response, "Concurrent requests")
        self.assertNotContains(response, "Not enforced")

    def test_org_admin_of_another_org_sees_only_their_own(self):
        response = self.as_user(self.other_org_admin).get(self.url)
        self.assertContains(response, "other-org")
        self.assertNotContains(response, "lab-org")
        self.assertNotContains(response, reverse("organization_limits", args=[self.org.id]))

    def test_unavailable_models_list_the_org_and_each_team_with_its_own(self):
        self.org.excluded_models = ["dall-e-2"]
        self.org.save()
        self.team.excluded_models = ["gpt-image-1"]
        self.team.save()
        response = self.as_user(self.org_admin).get(self.url)
        content = response.content.decode()
        everyone = content.split("Everyone in lab-org", 1)[1].split("</tr>", 1)[0]
        self.assertIn("dall-e-2", everyone)
        team_row = content.split(f'href="{reverse("team", args=[self.team.id])}"', 1)[1]
        self.assertIn("gpt-image-1", team_row.split("</tr>", 1)[0])
        self.assertNotIn("Robotics", content)

    @override_settings(AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST=["whisper-1"])
    def test_org_row_includes_server_defaults_only_while_the_org_merges_them(self):
        page = self.as_user(self.org_admin).get(self.url).content.decode()
        self.assertIn("whisper-1", page.split("Everyone in lab-org", 1)[1].split("</tr>", 1)[0])
        self.org.merge_exclusion_lists = False
        self.org.save()
        page = self.as_user(self.org_admin).get(self.url).content.decode()
        self.assertNotIn("whisper-1", page)

    def test_an_org_that_blocks_nothing_says_so_in_one_sentence(self):
        response = self.as_user(self.org_admin).get(self.url)
        self.assertContains(response, "Nothing is blocked for all of lab-org.")
        self.assertNotContains(response, "Everyone in lab-org</span>")
        self.assertNotContains(response, ">None<")


class OrganizationEditPermissionTest(OrganizationTestBase):
    def url(self, name: str, org: Org | None = None) -> str:
        return reverse(name, args=[(org or self.org).id])

    def test_refused_roles_get_404_and_change_nothing(self):
        before = self.stored(self.org)
        for user in (self.member, self.team_admin, self.solo, self.other_org_admin):
            for name, data in edits():
                with self.subTest(user=user.username, form=name):
                    client = self.as_user(user)
                    self.assertEqual(client.get(self.url(name)).status_code, HTTPStatus.NOT_FOUND)
                    response = client.post(self.url(name), data)
                    self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)
        self.assertEqual(self.stored(self.org), before)

    def test_org_admin_and_admin_may_edit(self):
        for user in (self.org_admin, self.far_admin):
            for name, data in edits():
                with self.subTest(user=user.username, form=name):
                    response = self.as_user(user).post(self.url(name), data)
                    self.assertRedirects(
                        response, reverse("organization"), fetch_redirect_response=False
                    )
        org = Org.objects.get(pk=self.org.pk)
        self.assertEqual(org.requests_per_minute, 600)
        self.assertIsNone(org.input_tokens_per_minute)
        self.assertEqual(org.output_tokens_per_minute, 5000)
        self.assertEqual(org.concurrent_requests_limit, 12)
        self.assertEqual(org.hourly_limit_multiplier, 30)
        self.assertEqual(org.excluded_models, ["gpt-4.1-nano"])

    def test_unknown_org_is_404(self):
        self.as_user(self.far_admin)
        url = reverse("organization_limits", args=[999_999])
        self.assertEqual(self.client.get(url).status_code, HTTPStatus.NOT_FOUND)


class OrganizationEditFormTest(OrganizationTestBase):
    def post(self, name: str, data: dict):
        url = reverse(name, args=[self.org.id])
        return self.as_user(self.org_admin).post(url, data)

    def test_bad_limits_are_form_errors_and_change_nothing(self):
        before = self.stored(self.org)
        for field, value in (
            ("requests_per_minute", "-1"),
            ("output_tokens_per_minute", "lots"),
            ("hourly_limit_multiplier", "61"),
            ("daily_limit_multiplier", "0"),
        ):
            with self.subTest(field=field, value=value):
                response = self.post("organization_limits", {**LIMITS, field: value})
                self.assertEqual(response.status_code, HTTPStatus.OK)
                self.assertTrue(response.context["form"].has_error(field))
        self.assertEqual(self.stored(self.org), before)

    def test_unknown_model_is_a_form_error_and_changes_nothing(self):
        before = self.stored(self.org)
        response = self.post("organization_availability", {"available_models": ["no-such-model"]})
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertTrue(response.context["form"].has_error("available_models"))
        self.assertEqual(self.stored(self.org), before)

    def form(self):
        url = reverse("organization_availability", args=[self.org.id])
        return self.as_user(self.org_admin).get(url).context["form"]

    def test_every_box_starts_checked(self):
        self.assertEqual(self.form()["available_models"].value(), list(model_names()))

    def test_saving_unchanged_keeps_what_is_stored(self):
        self.org.excluded_models = ["retired-model", "whisper-1", "gpt-4.1-nano"]
        self.org.save()
        with mcp_config():
            self.org.excluded_mcp_servers = ["retired-server", "aqueduct"]
            self.org.save()
            form = self.form()
            data = {name: form[name].value() for name in form.fields}
            data |= {
                f"{name}-shown": [value for value, _ in form.fields[name].choices]
                for name in ("available_models", "available_mcp_servers")
            }
            self.assertEqual(self.post("organization_availability", data).status_code, 302)
        org = Org.objects.get(pk=self.org.pk)
        self.assertEqual(org.excluded_models, ["retired-model", "whisper-1", "gpt-4.1-nano"])
        self.assertEqual(org.excluded_mcp_servers, ["retired-server", "aqueduct"])

    def test_clearing_a_box_makes_the_model_unavailable(self):
        self.post("organization_availability", available("dall-e-2", "whisper-1"))
        self.assertCountEqual(
            Org.objects.get(pk=self.org.pk).excluded_models, ["dall-e-2", "whisper-1"]
        )

    def test_names_missing_from_the_config_are_shown_unchecked_and_kept(self):
        self.org.excluded_models = ["retired-model", "gpt-4.1-nano"]
        self.org.save()
        form = self.as_user(self.org_admin).get(
            reverse("organization_availability", args=[self.org.id])
        )
        self.assertContains(form, "retired-model (not in the current config)")
        self.assertNotIn("retired-model", form.context["form"]["available_models"].value())
        response = self.post("organization_availability", available("gpt-4.1-nano", "dall-e-2"))
        self.assertEqual(response.status_code, HTTPStatus.FOUND)
        self.assertEqual(
            Org.objects.get(pk=self.org.pk).excluded_models,
            ["retired-model", "gpt-4.1-nano", "dall-e-2"],
        )

    def test_checking_a_name_missing_from_the_config_removes_it(self):
        self.org.excluded_models = ["retired-model"]
        self.org.save()
        data = available()
        data["available_models"].append("retired-model")
        self.post("organization_availability", data)
        self.assertEqual(Org.objects.get(pk=self.org.pk).excluded_models, [])

    def test_mcp_names_missing_from_the_config_are_kept(self):
        self.org.excluded_mcp_servers = ["retired-server"]
        self.org.save()
        with mcp_config():
            response = self.post("organization_availability", available(mcp=()))
        self.assertEqual(response.status_code, HTTPStatus.FOUND)
        self.assertEqual(Org.objects.get(pk=self.org.pk).excluded_mcp_servers, ["retired-server"])

    def test_empty_limits_clear_the_org_default(self):
        self.org.requests_per_minute = 600
        self.org.save()
        self.post("organization_limits", {**LIMITS, "requests_per_minute": ""})
        self.assertIsNone(Org.objects.get(pk=self.org.pk).requests_per_minute)

    def test_without_an_mcp_config_the_stored_unavailable_mcp_servers_stay(self):
        self.org.excluded_mcp_servers = ["kept"]
        self.org.save()
        with mcp_config("/nonexistent/mcp.json"):
            response = self.post("organization_availability", available("gpt-4.1-nano"))
        self.assertEqual(response.status_code, HTTPStatus.FOUND)
        self.assertEqual(Org.objects.get(pk=self.org.pk).excluded_mcp_servers, ["kept"])

    def test_mcp_servers_become_unavailable_for_the_org(self):
        with mcp_config():
            self.post("organization_availability", available("gpt-4.1-nano", mcp=("aqueduct",)))
        key = make_token(self.member, "laptop")
        self.assertTrue(enforced(key).mcp_server_excluded("aqueduct"))
        self.assertFalse(enforced(key).mcp_server_excluded("test-server"))


class OrganizationEditGatewayTest(OrganizationTestBase):
    def save(self, user, org: Org, data: dict):
        url = reverse("organization_availability", args=[org.id])
        return self.as_user(user).post(url, data)

    def test_unavailability_reaches_personal_and_service_account_keys_of_the_org(self):
        personal = make_token(self.member, "laptop")
        foreign = make_token(self.other_org_admin, "elsewhere")
        self.assertFalse(enforced(personal).model_excluded("gpt-4.1-nano"))

        self.save(self.org_admin, self.org, available("gpt-4.1-nano"))

        self.assertTrue(enforced(personal).model_excluded("gpt-4.1-nano"))
        self.assertTrue(enforced(self.pipeline_key).model_excluded("gpt-4.1-nano"))
        self.assertFalse(enforced(personal).model_excluded("whisper-1"))
        self.assertFalse(enforced(foreign).model_excluded("gpt-4.1-nano"))

        self.save(self.org_admin, self.org, available())
        self.assertFalse(enforced(personal).model_excluded("gpt-4.1-nano"))

    def test_foreign_org_admin_cannot_make_a_model_unavailable_for_this_org(self):
        personal = make_token(self.member, "laptop")
        response = self.save(self.other_org_admin, self.org, available("gpt-4.1-nano"))
        self.assertEqual(response.status_code, HTTPStatus.NOT_FOUND)
        self.assertFalse(enforced(personal).model_excluded("gpt-4.1-nano"))

    @override_settings(AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST=["whisper-1"])
    def test_clearing_merge_makes_the_server_defaults_available(self):
        personal = make_token(self.member, "laptop")
        self.assertTrue(enforced(personal).model_excluded("whisper-1"))
        self.save(self.org_admin, self.org, available())
        self.assertFalse(enforced(personal).model_excluded("whisper-1"))

    def test_merge_box_appears_only_with_server_defaults(self):
        url = reverse("organization_availability", args=[self.org.id])
        form = self.as_user(self.org_admin).get(url).context["form"]
        self.assertNotIn("merge_exclusion_lists", form.fields)
        with override_settings(AQUEDUCT_DEFAULT_MODEL_EXCLUSION_LIST=["whisper-1"]):
            response = self.as_user(self.org_admin).get(url)
        self.assertContains(response, "The server configuration makes whisper-1 unavailable")

    def test_models_page_hides_a_model_unavailable_in_the_org(self):
        self.save(self.org_admin, self.org, available("gpt-4.1-nano"))
        response = self.as_user(self.member).get(reverse("models"), {"model": "gpt-4.1-nano"})
        self.assertNotIn("gpt-4.1-nano", [card.name for card in response.context["catalog"].cards])
