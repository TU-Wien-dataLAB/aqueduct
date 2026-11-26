from datetime import timedelta
from pathlib import Path

from django.contrib.auth import get_user_model
from django.contrib.auth.models import Group
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from management.models import (
    Org,
    Request,
    ServiceAccount,
    Team,
    TeamMembership,
    Token,
    UserGroup,
    UserProfile,
)
from management.tests.helpers import join, make_token, make_user
from management.views.base import ORG_SESSION_KEY

User = get_user_model()

ROUTER_CONFIG = Path(__file__).resolve().parents[3] / "example_router_config.yaml"


class Matrix(TestCase):
    """north has teams north-a and north-b, south has south-a. Every user has a personal key
    and every team a service-account key. `stale_admin` and `stale_member` keep a membership
    in south-a, the state an org move leaves behind; `moved` created south-a's key in south
    and now lives in north; `orphan` has no profile."""

    @classmethod
    def setUpTestData(cls):
        cls.north = Org.objects.create(name="north")
        cls.south = Org.objects.create(name="south")
        cls.north_a = Team.objects.create(name="north-a", org=cls.north)
        cls.north_b = Team.objects.create(name="north-b", org=cls.north)
        cls.south_a = Team.objects.create(name="south-a", org=cls.south)

        cls.admin = make_user("admin", cls.north, UserGroup.ADMIN)
        cls.org_admin = make_user("org-admin", cls.north, UserGroup.ORG_ADMIN)
        cls.stale_admin = make_user("stale-admin", cls.north, UserGroup.ORG_ADMIN)
        cls.member = make_user("member", cls.north)
        cls.stale_member = make_user("stale-member", cls.north)
        cls.moved = make_user("moved", cls.north)
        cls.southerner = make_user("southerner", cls.south)
        cls.orphan = User.objects.create_user(username="orphan", email="orphan@example.com")
        join(cls.admin, cls.north_a)
        join(cls.org_admin, cls.north_a)
        join(cls.stale_admin, cls.south_a, is_admin=True)
        join(cls.member, cls.north_a)
        join(cls.member, cls.north_b, is_admin=True)
        join(cls.stale_member, cls.north_a)
        join(cls.stale_member, cls.south_a)
        join(cls.southerner, cls.south_a)

        for user in User.objects.all():
            make_token(user, f"{user.username}-laptop")
        for team, creator in (
            (cls.north_a, cls.member),
            (cls.north_b, cls.member),
            (cls.south_a, cls.moved),
        ):
            account = ServiceAccount.objects.create(name=f"{team.name}-bot", team=team)
            make_token(creator, f"{team.name}-bot", service_account=account)

    def profile(self, user) -> UserProfile:
        return UserProfile.objects.select_related("org", "user").get(user=user)

    def names(self, tokens) -> set[str]:
        return set(tokens.values_list("name", flat=True))


class KeyAccessTest(Matrix):
    def test_each_rank_reaches_its_keys(self):
        everything = self.names(Token.objects.all())
        north = everything - {"southerner-laptop", "south-a-bot", "orphan-laptop"}
        south = {"southerner-laptop", "south-a-bot"}
        bots = {"north-a-bot", "north-b-bot", "south-a-bot"}
        admin, org_admin = self.profile(self.admin), self.profile(self.org_admin)
        member = self.profile(self.member)
        for label, tokens, expected in (
            ("admin, every org", admin.visible_tokens(), everything),
            ("admin, south", admin.visible_tokens(self.south), south),
            ("admin, own", admin.own_tokens(), {"admin-laptop"}),
            ("admin, mine", admin.my_tokens(), {"admin-laptop", "north-a-bot"}),
            ("admin, manage", admin.manageable_tokens(), {"admin-laptop", *bots}),
            ("org admin", org_admin.visible_tokens(), north),
            ("org admin, another org asked", org_admin.visible_tokens(self.south), north),
            (
                "org admin, manage",
                org_admin.manageable_tokens(),
                {"org-admin-laptop", "north-a-bot", "north-b-bot"},
            ),
            ("member", member.visible_tokens(), {"member-laptop", "north-a-bot", "north-b-bot"}),
            ("member, mine", member.my_tokens(), {"member-laptop", "north-a-bot", "north-b-bot"}),
            ("member, manage", member.manageable_tokens(), {"member-laptop", "north-b-bot"}),
            ("member, own", member.own_tokens(), {"member-laptop"}),
            (
                "stale member",
                self.profile(self.stale_member).visible_tokens(),
                {"stale-member-laptop", "north-a-bot"},
            ),
            ("moved", self.profile(self.moved).visible_tokens(), {"moved-laptop"}),
        ):
            with self.subTest(label):
                self.assertEqual(self.names(tokens), expected)

    def test_the_key_sets_nest(self):
        for user in (self.admin, self.org_admin, self.member, self.stale_member, self.moved):
            profile = self.profile(user)
            visible = self.names(profile.visible_tokens())
            with self.subTest(user=user.username):
                own, mine = self.names(profile.own_tokens()), self.names(profile.my_tokens())
                manageable = self.names(profile.manageable_tokens())
                self.assertLessEqual(own, mine)
                self.assertLessEqual(mine, visible)
                self.assertLessEqual(own, manageable)
                self.assertLessEqual(manageable, visible)
                for org in (self.north, self.south):
                    self.assertLessEqual(self.names(profile.visible_tokens(org)), visible)

    def test_a_stale_membership_grants_nothing(self):
        for user in (self.stale_admin, self.stale_member):
            profile = self.profile(user)
            with self.subTest(user=user.username):
                for tokens in (
                    profile.visible_tokens(),
                    profile.visible_tokens(self.south),
                    profile.my_tokens(),
                    profile.manageable_tokens(),
                ):
                    self.assertNotIn("south-a-bot", self.names(tokens))
                for teams in (
                    profile.member_teams(),
                    profile.member_teams(admin=True),
                    profile.visible_teams(),
                    profile.managed_teams(),
                ):
                    self.assertNotIn(self.south_a, teams)
                self.assertFalse(profile.is_team_admin(self.south_a))

    def test_requests_follow_the_visible_keys(self):
        sent = timezone.now() - timedelta(hours=1)
        Request.objects.bulk_create(
            Request(token=token, path=f"/{token.name}", timestamp=sent)
            for token in Token.objects.all()
        )
        admin = self.profile(self.admin)
        self.assertEqual(admin.visible_requests().count(), Token.objects.count())
        for user, org in ((self.admin, self.south), (self.org_admin, None), (self.member, None)):
            profile = self.profile(user)
            with self.subTest(user=user.username, org=org):
                self.assertEqual(
                    set(profile.visible_requests(org).values_list("token__name", flat=True)),
                    self.names(profile.visible_tokens(org)),
                )


class TeamAccessTest(Matrix):
    def test_each_rank_reaches_its_teams(self):
        every = {self.north_a, self.north_b, self.south_a}
        north = {self.north_a, self.north_b}
        admin, org_admin = self.profile(self.admin), self.profile(self.org_admin)
        member = self.profile(self.member)
        for label, teams, expected in (
            ("admin, every org", admin.visible_teams(), every),
            ("admin, south", admin.visible_teams(self.south), {self.south_a}),
            ("admin, manage", admin.managed_teams(), every),
            ("admin, member", admin.member_teams(), {self.north_a}),
            ("org admin", org_admin.visible_teams(), north),
            ("org admin, another org asked", org_admin.visible_teams(self.south), north),
            ("org admin, manage", org_admin.managed_teams(), north),
            ("member", member.visible_teams(), north),
            ("member, team admin", member.member_teams(admin=True), {self.north_b}),
            ("member, manage", member.managed_teams(), {self.north_b}),
            ("southerner", self.profile(self.southerner).visible_teams(), {self.south_a}),
        ):
            with self.subTest(label):
                self.assertEqual(set(teams), expected)

    def test_team_admins_manage_but_only_org_admins_administer(self):
        member, org_admin = self.profile(self.member), self.profile(self.org_admin)
        self.assertTrue(member.is_team_admin(self.north_b))
        self.assertFalse(member.is_team_admin(self.north_a))
        self.assertFalse(member.is_org_admin(self.north_b.org))
        self.assertTrue(org_admin.is_team_admin(self.north_a))
        self.assertFalse(org_admin.is_team_admin(self.south_a))
        self.assertTrue(self.profile(self.admin).is_team_admin(self.south_a))

    def test_another_members_admin_flag_makes_no_one_else_a_team_admin(self):
        TeamMembership.objects.filter(
            user_profile__user=self.stale_member, team=self.north_a
        ).update(is_admin=True)
        member = self.profile(self.member)
        self.assertEqual(set(member.member_teams(admin=True)), {self.north_b})
        self.assertEqual(set(member.managed_teams()), {self.north_b})
        self.assertFalse(member.is_team_admin(self.north_a))

    def test_each_rank_administers_its_orgs(self):
        for user, expected in (
            (self.admin, {self.north, self.south}),
            (self.org_admin, {self.north}),
            (self.stale_admin, {self.north}),
            (self.member, set()),
        ):
            profile = self.profile(user)
            with self.subTest(user=user.username):
                self.assertEqual(set(profile.administered_orgs()), expected)
                self.assertEqual(
                    {o for o in (self.north, self.south) if profile.is_org_admin(o)}, expected
                )


class CrossOrgPagesTest(Matrix):
    def test_a_stale_membership_opens_no_team_page_and_lists_no_key(self):
        for user in (self.stale_admin, self.stale_member):
            self.client.force_login(user)
            with self.subTest(user=user.username):
                response = self.client.get(reverse("team", args=[self.south_a.id]))
                self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
                keys = self.client.get(reverse("tokens"))
                self.assertNotContains(keys, "south-a-bot")

    def test_an_org_admin_cannot_delete_another_orgs_team(self):
        self.client.force_login(self.org_admin)
        response = self.client.post(reverse("team_delete", args=[self.south_a.id]))
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
        self.assertTrue(Team.objects.filter(pk=self.south_a.pk).exists())

    def test_an_org_admin_cannot_appoint_team_admins_in_another_org(self):
        self.client.force_login(self.org_admin)
        southerner = self.southerner.profile
        response = self.client.post(
            reverse("team_admins_manage", args=[self.south_a.id]),
            {"profile_id": southerner.id, "action": "add"},
        )
        self.assertRedirects(response, reverse("org"), fetch_redirect_response=False)
        self.assertFalse(
            TeamMembership.objects.get(user_profile=southerner, team=self.south_a).is_admin
        )


@override_settings(LITELLM_ROUTER_CONFIG_FILE_PATH=ROUTER_CONFIG)
class UsagePageTest(Matrix):
    def usage(self, user, **params):
        self.client.force_login(user)
        response = self.client.get(reverse("usage"), params)
        self.assertEqual(response.status_code, 200)
        return response

    def key(self, name: str) -> Token:
        return Token.objects.get(name=name)

    def test_a_user_cannot_select_a_key_they_may_not_see(self):
        for name in ("member-laptop", "north-a-bot"):
            with self.subTest(key=name):
                page = self.usage(self.moved, token=self.key(name).id)
                self.assertIsNone(page.context["selected_token"])
                self.assertNotContains(page, f'for "{name}"')

    def test_a_user_selects_their_own_key_and_their_teams_key(self):
        for name in ("member-laptop", "north-a-bot"):
            with self.subTest(key=name):
                page = self.usage(self.member, token=self.key(name).id)
                self.assertEqual(page.context["selected_token"], self.key(name))
                self.assertContains(page, f'for "{name}"')

    def test_admins_select_any_key_they_can_see(self):
        for user, params, name in (
            (self.org_admin, {}, "member-laptop"),
            (self.org_admin, {}, "north-b-bot"),
            (self.admin, {"org": "all"}, "south-a-bot"),
            (self.admin, {"org": self.south.id}, "southerner-laptop"),
        ):
            with self.subTest(user=user.username, key=name):
                page = self.usage(user, token=self.key(name).id, **params)
                self.assertEqual(page.context["selected_token"], self.key(name))
        page = self.usage(self.org_admin, token=self.key("south-a-bot").id)
        self.assertIsNone(page.context["selected_token"])

    def test_a_non_numeric_parameter_selects_nothing(self):
        for user in (self.member, self.admin):
            with self.subTest(user=user.username):
                page = self.usage(user, token="²", org="²")
                self.assertIsNone(page.context["selected_token"])
                self.assertEqual(page.context["selected_org"], self.north)


class OrgSwitchTest(Matrix):
    def switch(self, org: Org):
        return self.client.post(reverse("org_switch"), {"org_id": org.id})

    def test_a_switch_keeps_the_admins_org_and_team_memberships(self):
        self.client.force_login(self.admin)
        self.assertRedirects(self.switch(self.south), reverse("org"))
        profile = self.profile(self.admin)
        self.assertEqual(profile.org, self.north)
        self.assertEqual(set(profile.member_teams()), {self.north_a})
        self.assertTrue(
            TeamMembership.objects.filter(user_profile=profile, team=self.north_a).exists()
        )
        self.assertEqual(self.client.session[ORG_SESSION_KEY], self.south.id)

    def test_the_teams_page_shows_the_chosen_org(self):
        self.client.force_login(self.admin)
        self.switch(self.south)
        page = self.client.get(reverse("org"))
        self.assertEqual(page.context["org_object"], self.south)
        self.assertEqual(set(page.context["teams"]), {self.south_a})
        self.assertContains(page, '<span class="text-3xl font-semibold ml-1">south</span>')
        self.assertContains(page, f'<option value="{self.south.id}" selected>', html=False)

    def test_a_new_team_lands_in_the_chosen_org(self):
        self.client.force_login(self.admin)
        self.switch(self.south)
        self.client.post(reverse("team_create"), {"name": "vision"})
        self.assertEqual(Team.objects.get(name="vision").org, self.south)

    def test_non_admins_cannot_switch_and_ignore_a_left_over_choice(self):
        self.client.force_login(self.org_admin)
        self.switch(self.south)
        self.assertNotIn(ORG_SESSION_KEY, self.client.session)
        session = self.client.session
        session[ORG_SESSION_KEY] = self.south.id
        session.save()
        page = self.client.get(reverse("org"))
        self.assertEqual(page.context["org_object"], self.north)
        self.assertEqual(set(page.context["teams"]), {self.north_a, self.north_b})


class GroupTest(TestCase):
    def setUp(self):
        self.profile = make_user("ana", Org.objects.create(name="north")).profile

    def test_the_highest_group_decides(self):
        self.profile.user.groups.set(Group.objects.filter(name__in=list(UserGroup.values)))
        self.assertEqual(self.profile.group, UserGroup.ADMIN)

    def test_a_user_without_a_group_administers_nothing(self):
        self.profile.user.groups.clear()
        self.assertFalse(self.profile.is_admin())
        self.assertFalse(self.profile.is_org_admin(self.profile.org))

    def test_the_group_is_read_once_per_profile(self):
        profile = UserProfile.objects.select_related("user", "org").get(pk=self.profile.pk)
        with self.assertNumQueries(1):
            for _ in range(3):
                profile.is_admin()
                profile.is_org_admin(profile.org)
                self.assertIs(profile.group, UserGroup.USER)

    def test_setting_the_group_forgets_the_one_read(self):
        self.assertFalse(self.profile.is_admin())
        self.profile.group = UserGroup.ADMIN
        self.assertTrue(self.profile.is_admin())
