from pathlib import Path

from django.contrib.auth import get_user_model
from django.db import connection, transaction
from django.test import TestCase, override_settings
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from management.auth import OIDCBackend
from management.models import Org, ServiceAccount, Team, TeamMembership, UserGroup, UserProfile
from management.tests.helpers import join, make_token, make_user, seed_active_config
from management.tests.test_oauth_team_creation import BASE_SETTINGS

User = get_user_model()

ROUTER_CONFIG = Path(__file__).resolve().parents[3] / "example_router_config.yaml"

ORG_THEN_TEAMS = """\
class MyConfig(ConfigSnippet):
    def org_name(self, claims):
        return claims["groups"][0]

    def team_names(self, claims):
        return claims["groups"][1:]
"""


class OrgMoveTest(TestCase):
    def setUp(self):
        self.north = Org.objects.create(name="north")
        self.south = Org.objects.create(name="south")
        self.north_team = Team.objects.create(name="E123", org=self.north)
        self.user = make_user("ana", self.north)
        join(self.user, self.north_team, is_admin=True)

    def teams_of(self, user) -> set[Team]:
        return set(Team.objects.filter(member_profiles__user=user))

    def test_an_org_move_ends_the_memberships_of_the_old_org_and_logs_them(self):
        south_team = Team.objects.create(name="Robotics", org=self.south)
        pipeline = ServiceAccount.objects.create(name="pipeline", team=self.north_team)
        make_token(self.user, "pipeline", service_account=pipeline)
        profile = UserProfile.objects.get(user=self.user)
        profile.org = self.south
        with self.assertLogs("aqueduct", "INFO") as logs:
            profile.save()
        join(self.user, south_team)
        profile.save()
        self.assertEqual(self.teams_of(self.user), {south_team})
        team = self.north_team.pk
        self.assertEqual(
            logs.output,
            [
                f"INFO:aqueduct:Org move of ana@example.com ends their membership in team {team} "
                "(E123), admin: True.",
                f"WARNING:aqueduct:ana@example.com left team {team} but owns its service account "
                f"{pipeline.pk} (pipeline); transfer it.",
            ],
        )

    def test_a_save_without_an_org_move_deletes_nothing(self):
        profile = UserProfile.objects.get(user=self.user)
        with CaptureQueriesContext(connection) as queries:
            profile.save()
            profile.org = self.south
            profile.save(update_fields=["requests_per_minute"])
        self.assertFalse([q for q in queries if q["sql"].startswith("DELETE")])
        self.assertEqual(self.teams_of(self.user), {self.north_team})

    def test_a_save_compares_with_the_stored_org_not_the_loaded_one(self):
        stale = UserProfile.objects.get(user=self.user)
        UserProfile.objects.filter(user=self.user).update(org=self.south)
        stale.refresh_from_db()
        stale.save()
        UserProfile.objects.only("id", "user").get(user=self.user).save()
        self.assertEqual(self.teams_of(self.user), {self.north_team})

    def test_a_move_retried_after_a_rollback_still_ends_the_old_memberships(self):
        profile = UserProfile.objects.get(user=self.user)
        profile.org = self.south
        with self.assertRaises(RuntimeError), transaction.atomic():
            profile.save()
            raise RuntimeError
        profile.save()
        self.assertEqual(self.teams_of(self.user), set())

    @override_settings(**BASE_SETTINGS)
    def test_a_sign_in_from_another_org_moves_the_user_with_their_oauth_teams(self):
        seed_active_config(ORG_THEN_TEAMS)
        self.north_team.oauth_group_name = "E123"
        self.north_team.save()
        south_team = Team.objects.create(name="E123", org=self.south, oauth_group_name="E123")

        OIDCBackend().update_user(self.user, {"groups": ["south", "E123"]})

        self.assertEqual(UserProfile.objects.get(user=self.user).org, self.south)
        self.assertEqual(self.teams_of(self.user), {south_team})


class OrgSwitchTest(TestCase):
    def test_a_platform_admin_switching_org_keeps_their_memberships(self):
        north, south = Org.objects.create(name="north"), Org.objects.create(name="south")
        team = Team.objects.create(name="NLP", org=north)
        admin = make_user("root", north, UserGroup.ADMIN)
        join(admin, team, is_admin=True)
        self.client.force_login(admin)

        for org in (south, "all", north):
            self.client.post(reverse("org_switch"), {"org_id": getattr(org, "pk", org)})

            self.assertEqual(UserProfile.objects.get(user=admin).org, north)
            self.assertEqual(set(admin.profile.teams.all()), {team})


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG)
class TeamAdminTest(TestCase):
    def setUp(self):
        self.north = Org.objects.create(name="north")
        self.south = Org.objects.create(name="south")
        self.team = Team.objects.create(name="NLP", org=self.north)
        self.member = make_user("member", self.north)
        self.foreigner = make_user("foreigner", self.south)
        join(self.member, self.team)
        admin = make_user("admin", self.north)
        admin.is_staff = admin.is_superuser = True
        admin.save()
        self.client.force_login(admin)
        self.url = reverse("admin:management_team_change", args=[self.team.pk])

    def post(self, **changes):
        """Submit the change form as rendered, with `changes` applied."""
        page = self.client.get(self.url).context
        forms = [page["adminform"].form]
        for formset in page["inline_admin_formsets"]:
            forms += [formset.formset.management_form, *formset.formset.forms]
        data = {}
        for form in forms:
            for name in form.fields:
                value = form[name].value()
                if value is None or value is False:
                    continue
                data[form.add_prefix(name)] = (
                    [str(v) for v in value] if isinstance(value, list) else str(value)
                )
        data.update(changes)
        return self.client.post(self.url, data)

    def test_refuses_a_member_from_another_org(self):
        response = self.post(**{"teammembership_set-1-user_profile": self.foreigner.profile.pk})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "foreigner@example.com is in another organization.")
        self.assertEqual(set(self.team.member_profiles.all()), {self.member.profile})

    def test_refuses_to_move_a_team_with_members_to_another_org(self):
        response = self.post(org=self.south.pk)
        self.assertContains(response, "member@example.com is in another organization.")
        self.team.refresh_from_db()
        self.assertEqual(self.team.org, self.north)

    def test_moves_a_team_once_its_members_are_removed(self):
        response = self.post(org=self.south.pk, **{"teammembership_set-0-DELETE": "on"})
        self.assertEqual(response.status_code, 302)
        self.team.refresh_from_db()
        self.assertEqual(self.team.org, self.south)
        self.assertFalse(TeamMembership.objects.filter(team=self.team).exists())
